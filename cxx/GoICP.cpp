#include "GoICP.hpp"

GoICP::GoICP(float mseThresh, float rotMinX, float rotMinY, float rotMinZ,
             float rotWidth, float transMinX, float transMinY, float transMinZ,
             float transWidth, float trimFraction, float expandFactor, uint32_t div)
    : icp(),
      mseThresh(mseThresh), trimFraction(trimFraction), doTrim(false),
      expandFactor(expandFactor), div(div)
{
    initNodeRot.a = rotMinX;
    initNodeRot.b = rotMinY;
    initNodeRot.c = rotMinZ;
    initNodeRot.w = rotWidth;
    initNodeTrans.x = transMinX;
    initNodeTrans.y = transMinY;
    initNodeTrans.z = transMinZ;
    initNodeTrans.w = transWidth;
    initNodeRot.l = 0;
    initNodeRot.lb = 0;
    initNodeTrans.lb = 0;
}

void GoICP::SetSource(PointCloudPtr src)
{
    pData = src;
}

// Cloud setting and DT building (make sure to call this before calling Register())
void GoICP::SetTarget(PointCloudPtr tgt)
{
    pModel = tgt;
    icp.setInputTarget(tgt);
    ldt = std::make_unique<LinearDT>(pModel, expandFactor, div);
}

const int GoICP::GetModelSize()
{
    return pModel->points.size();
}

const int GoICP::GetDataSize()
{
    return pData->points.size();
}

const Matrix4f GoICP::GetOptMat()
{
    return optMat;
}

// Run ICP and calculate sum squared L2 error
float GoICP::runICP(Matrix4f &trans_now)
{
    icp.setInputSource(pData);
    auto out = boost::make_shared<PointCloud>();
    icp.align(*out, trans_now);
    trans_now = icp.getFinalTransformation();

    // Transform point cloud and use DT to determine the L2 error
    std::vector<float> error(out->size());
    tbb::parallel_for(0, int(out->size()), [&](int i) {
        Point3f &ptemp = out->points[i];
        float dis = ldt->Evaluate(ptemp);
        error[i] = dis * dis;
    });

    float SSE = 0.0f;
    for (int i = 0; i < inlierNum; i++)
        SSE += error[i];

    return SSE;
}

void GoICP::initialize()
{
    int i, j;
    float sigma, maxAngle;
    auto normData = std::vector<float>(pData->points.size());

    // Precompute the rotation uncertainty distance (maxRotDis) for each point in the data
    // and each level of rotation subcube
    // Calculate L2 norm of each point in data cloud to origin
    for (i = 0; i < pData->size(); i++)
    {
        normData[i] = std::sqrt(SQ(pData->points[i].x) + SQ(pData->points[i].y) + SQ(pData->points[i].z));
    }

    maxRotDis = new float *[MAXROTLEVEL];
    for (i = 0; i < MAXROTLEVEL; i++)
    {
        maxRotDis[i] = (float *)malloc(sizeof(float *) * pData->points.size());

        sigma = initNodeRot.w / std::pow(2.0, i) / 2.0; // Half-side length of each level of rotation subcube
        maxAngle = SQRT3 * sigma;

        if (maxAngle > PI)
            maxAngle = PI;
        for (j = 0; j < pData->points.size(); j++)
            maxRotDis[i][j] = 2 * std::sin(maxAngle / 2) * normData[j];
    }

    // Temporary Variable
    // we declare it here because we don't want these space to be allocated and deallocated
    // again and again each time inner BNB runs.
    minDis = std::vector<float>(pData->points.size());

    // Initialise so-far-best rotation and translation nodes
    optNodeRot = initNodeRot;
    optNodeTrans = initNodeTrans;
    // Initialise so-far-best rotation and translation matrix
    optMat = Eigen::Matrix4f::Identity();

    // For untrimmed ICP, use all points, otherwise only use inlierNum points
    inlierNum = pData->size();
    sseThresh = mseThresh * inlierNum;
}

void GoICP::clear()
{
    for (int i = 0; i < MAXROTLEVEL; i++)
        delete (maxRotDis[i]);
    delete (maxRotDis);
}

// Inner Branch-and-Bound, iterating over the translation space
float GoICP::innerBnB(float *maxRotDisL, TranslationNode *nodeTransOut,
                      const std::vector<Point3f> &pDataTemp)
{
    int j;
    float transX, transY, transZ;
    float lb, ub, optErrorT;
    float maxTransDis;
    TranslationNode nodeTrans, nodeTransParent;
    std::priority_queue<TranslationNode> queueTrans;

    // Set optimal translation error to overall so-far optimal error
    // Investigating translation nodes that are sub-optimal overall is redundant
    optErrorT = optError;

    // Push top-level translation node into the priority queue
    queueTrans.push(initNodeTrans);

    while (true)
    {
        if (queueTrans.empty())
            break;

        // std::cout << '.';
        // fflush(stdout);

        nodeTransParent = queueTrans.top();
        queueTrans.pop();

        if (optErrorT - nodeTransParent.lb < sseThresh)
        {
            break;
        }

        nodeTrans.w = nodeTransParent.w / 2;
        maxTransDis = SQRT3 / 2.0 * nodeTrans.w;

        for (j = 0; j < 8; j++)
        {
            nodeTrans.x = nodeTransParent.x + (j & 1) * nodeTrans.w;
            nodeTrans.y = nodeTransParent.y + (j >> 1 & 1) * nodeTrans.w;
            nodeTrans.z = nodeTransParent.z + (j >> 2 & 1) * nodeTrans.w;

            transX = nodeTrans.x + nodeTrans.w / 2;
            transY = nodeTrans.y + nodeTrans.w / 2;
            transZ = nodeTrans.z + nodeTrans.w / 2;

            struct ReduceElement
            {
                float sqr;
                float sqr_subTransDis;
            };

            auto minDistSq = std::vector<float>(pData->size());
            auto subTransDistSq = std::vector<float>(pData->size());
            auto minDisSqr = std::make_unique<ReduceElement[]>(pData->points.size());

            // For each data point, calculate the distance to it's closest point in the model cloud
            tbb::parallel_for(0, int(pData->points.size()), [&](int i) {
                // Find distance between transformed point and closest point in model set
                // ||R_r0 * x + t0 - y||
                // pDataTemp is the data points rotated by R0
                minDis[i] = ldt->Evaluate(Point3f(pDataTemp[i].data[0] + transX,
                                                  pDataTemp[i].data[1] + transY,
                                                  pDataTemp[i].data[2] + transZ));

                // Subtract the rotation uncertainty radius if calculating the rotation lower bound
                // maxRotDisL == NULL when calculating the rotation upper bound
                if (maxRotDisL)
                    minDis[i] = std::max(minDis[i] - maxRotDisL[i], 0.0f);

                minDistSq[i] = minDis[i] * minDis[i];
                subTransDistSq[i] = SQ(std::max(minDis[i] - maxTransDis, 0.0f));
            });

            // For each data point, find the incremental upper and lower bounds
            auto ub = 0.f, lb = 0.f;
            for (auto i = 0u; i < pData->size(); i++)
            {
                ub += minDistSq[i];
                lb += subTransDistSq[i];
            }

            // If upper bound is better than best, update optErrorT and optTransOut (optimal
            // translation node)
            if (ub < optErrorT)
            {
                optErrorT = ub;
                if (nodeTransOut)
                    *nodeTransOut = nodeTrans;
            }

            // Remove subcube from queue if lb is bigger than optErrorT
            if (lb >= optErrorT)
            {
                //discard
                continue;
            }

            nodeTrans.ub = ub;
            nodeTrans.lb = lb;
            queueTrans.push(nodeTrans);
        }
    }

    return optErrorT;
}

float GoICP::outerBnB()
{
    int i, j;
    RotationNode nodeRot, nodeRotParent;
    TranslationNode nodeTrans;
    float v1, v2, v3, t, ct, ct2, st, st2;
    float tmp121, tmp122, tmp131, tmp132, tmp231, tmp232;
    float R11, R12, R13, R21, R22, R23, R31, R32, R33;
    float lb, ub, error;
    clock_t clockBeginICP;
    std::priority_queue<RotationNode> queueRot;
    auto pDataTemp = std::vector<Point3f>(pData->points.size());

    // Calculate Initial Error
    optError = 0;

    for (i = 0; i < pData->points.size(); i++)
    {
        minDis[i] = ldt->Evaluate(Point3f(pData->points[i]));
    }
    for (i = 0; i < inlierNum; i++)
    {
        optError += minDis[i] * minDis[i];
    }
    // std::cout << "Error*: " << optError << " (Init)" << '\n';

    // Temporary matrix storeing both input and output of icp (through citation, &).
    // Because the approximating nature of DT, ICP result may return  an error slightly bigger than
    // formal optimal one, we cannot directly let ICP change the optimal matrix.
    Matrix4f mat = optMat;

    // Run ICP from initial state
    clockBeginICP = clock();
    error = runICP(mat);
    if (error < optError)
    {
        optError = error;
        optMat = mat;
        // std::cout << "Error*: " << error << " (ICP " << (double)(clock() - clockBeginICP) / CLOCKS_PER_SEC << "s)" << std::endl;
        // std::cout << "ICP-ONLY Affine Matrix:" << '\n';
        // std::cout << mat.matrix() << '\n';
    }

    // Push top-level rotation node into priority queue
    queueRot.push(initNodeRot);

    // Keep exploring rotation space until convergence is achieved
    long long count = 0;
    while (true)
    {
        if (queueRot.empty())
        {
            // std::cout << "Rotation Queue Empty" << '\n';
            // std::cout << "Error*: " << optError << ", LB: " << lb << '\n';
            break;
        }

        // std::cout << 'o';
        fflush(stdout);

        // Access rotation cube with lowest lower bound...
        nodeRotParent = queueRot.top();
        // ...and remove it from the queue
        queueRot.pop();

        // Exit if the optError is less than or equal to the lower bound plus a small epsilon
        if ((optError - nodeRotParent.lb) <= sseThresh)
        {
            // std::cout << "Error*: " << optError << ", LB: " << nodeRotParent.lb << ", epsilon: "
                    //   << sseThresh << '\n';
            break;
        }

        if (count++ % 10 == 0)
        {
            // printf("LB=%f  L=%d  ", nodeRotParent.lb, nodeRotParent.l);
            // std::cout << "optError: " << optError << ", LB: " << lb << '\n';
        }

        // Subdivide rotation cube into octant subcubes and calculate upper and lower bounds for each
        nodeRot.w = nodeRotParent.w / 2;
        nodeRot.l = nodeRotParent.l + 1;
        // For each subcube,
        for (j = 0; j < 8; j++)
        {
            // Calculate the smallest rotation across each dimension
            nodeRot.a = nodeRotParent.a + (j & 1) * nodeRot.w;
            nodeRot.b = nodeRotParent.b + (j >> 1 & 1) * nodeRot.w;
            nodeRot.c = nodeRotParent.c + (j >> 2 & 1) * nodeRot.w;

            // Find the subcube centre
            v1 = nodeRot.a + nodeRot.w / 2;
            v2 = nodeRot.b + nodeRot.w / 2;
            v3 = nodeRot.c + nodeRot.w / 2;

            // Skip subcube if it is completely outside the rotation PI-ball
            if (sqrt(v1 * v1 + v2 * v2 + v3 * v3) - SQRT3 * nodeRot.w / 2 > PI)
            {
                continue;
            }

            // Convert angle-axis rotation into a rotation matrix
            t = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
            if (t > 0)
            {
                v1 /= t;
                v2 /= t;
                v3 /= t;

                ct = cos(t);
                ct2 = 1 - ct;
                st = sin(t);
                st2 = 1 - st;

                tmp121 = v1 * v2 * ct2;
                tmp122 = v3 * st;
                tmp131 = v1 * v3 * ct2;
                tmp132 = v2 * st;
                tmp231 = v2 * v3 * ct2;
                tmp232 = v1 * st;

                R11 = ct + v1 * v1 * ct2;
                R12 = tmp121 - tmp122;
                R13 = tmp131 + tmp132;
                R21 = tmp121 + tmp122;
                R22 = ct + v2 * v2 * ct2;
                R23 = tmp231 - tmp232;
                R31 = tmp131 - tmp132;
                R32 = tmp231 + tmp232;
                R33 = ct + v3 * v3 * ct2;

                // Rotate data points by subcube rotation matrix
                tbb::parallel_for(0, int(pData->points.size()), [&](int i) {
                    Point3f &p = pData->points[i];
                    pDataTemp[i].x = R11 * p.x + R12 * p.y + R13 * p.z;
                    pDataTemp[i].y = R21 * p.x + R22 * p.y + R23 * p.z;
                    pDataTemp[i].z = R31 * p.x + R32 * p.y + R33 * p.z;
                });
            }
            // If t == 0, the rotation angle is 0 and no rotation is required
            else
                tbb::parallel_for(0, int(pData->points.size()), [&](int i) {
                    pDataTemp[i] = pData->points[i];
                });

            // Upper Bound
            // Run Inner Branch-and-Bound to find rotation upper bound
            // Calculates the rotation upper bound by finding the translation upper bound for a given
            // rotation, assuming that the rotation is known (zero rotation uncertainty radius)
            ub = innerBnB(NULL /*Rotation Uncertainty Radius*/, &nodeTrans, pDataTemp);

            // If the upper bound is the best so far, run ICP
            if (ub < optError)
            {
                // Update optimal error and rotation/translation nodes
                optError = ub;
                optNodeRot = nodeRot;
                optNodeTrans = nodeTrans;

                optMat(0, 0) = R11;
                optMat(0, 1) = R12;
                optMat(0, 2) = R13;
                optMat(1, 0) = R21;
                optMat(1, 1) = R22;
                optMat(1, 2) = R23;
                optMat(2, 0) = R31;
                optMat(2, 1) = R32;
                optMat(2, 2) = R33;
                optMat(0, 3) = optNodeTrans.x + optNodeTrans.w / 2;
                optMat(1, 3) = optNodeTrans.y + optNodeTrans.w / 2;
                optMat(2, 3) = optNodeTrans.z + optNodeTrans.w / 2;

                // std::cout << "Error*: " << optError << '\n';

                // Run ICP
                clockBeginICP = clock();
                mat = optMat;
                error = runICP(mat);
                // Our ICP implementation uses kdtree for closest distance computation which is slightly
                // different from DT approximation, thus it's possible that ICP failed to decrease the
                // DT error. This is no big deal as the difference should be very small.
                if (error < optError)
                {
                    optError = error;
                    optMat = mat;
                    // std::cout << "Error*: " << error << "(ICP " << (double)(clock() - clockBeginICP) / CLOCKS_PER_SEC << "s)" << '\n';
                }

                // Discard all rotation nodes with high lower bounds in the queue
                std::priority_queue<RotationNode> queueRotNew;
                while (!queueRot.empty())
                {
                    RotationNode node = queueRot.top();
                    queueRot.pop();
                    if (node.lb < optError)
                        queueRotNew.push(node);
                    else
                        break;
                }
                queueRot = queueRotNew;
            }

            // Lower Bound
            // Run Inner Branch-and-Bound to find rotation lower bound
            // Calculates the rotation lower bound by finding the translation upper bound for a given
            // rotation, assuming that the rotation is uncertain (a positive rotation uncertainty radius)
            // Pass an array of rotation uncertainties for every point in data cloud at this level
            lb = innerBnB(maxRotDis[nodeRot.l], nullptr /*Translation Node*/, pDataTemp);

            // If the best error so far is less than the lower bound, remove the rotation subcube from
            // the queue
            if (lb >= optError)
            {
                continue;
            }

            // Update node and put it in queue
            nodeRot.ub = ub;
            nodeRot.lb = lb;
            queueRot.push(nodeRot);
        }
    }

    return optError;
}

float GoICP::Register()
{
    initialize();
    outerBnB();
    clear();
    return optError;
}
