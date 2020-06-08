#include "GoICP.hpp"
#include "Timer.hpp"

#include <random>
#include <pcl/registration/ia_fpcs.h>
#include <H5Cpp.h>

static constexpr auto TEST_SIZE = 50;

std::vector<PointCloudPtr> loadData(const char *path)
{
    // Open test dataset
    H5::H5File file(path, H5F_ACC_RDONLY);
    auto dataset = file.openDataSet("data");

    // Allocate buffer space
    hsize_t dims[3]{TEST_SIZE, 2048, 3};
    auto buf = std::make_unique<float[]>(dims[0] * dims[1] * dims[2]);

    // Speficy memory and file space
    H5::DataSpace mem_space(3, dims);
    auto file_space = dataset.getSpace();
    hsize_t start[3]{0, 0, 0};
    file_space.selectHyperslab(H5S_SELECT_SET, dims, start);

    // Read to buffer
    dataset.read(buf.get(), H5::PredType::IEEE_F32LE, mem_space, file_space);

    // Close file
    dataset.close();
    file.close();

    // Create point clouds
    std::vector<PointCloudPtr> clouds;
    clouds.reserve(TEST_SIZE);
    auto cloud_size = dims[1] * dims[2];
    for (auto i = 0; i < TEST_SIZE; i++)
    {
        auto cloud = boost::make_shared<PointCloud>();
        cloud->reserve(dims[1]);
        for (auto j = 0; j < dims[1]; j++)
        {
            auto ptr = &buf[i * cloud_size + j * dims[2]];
            cloud->push_back(Point3f(ptr[0], ptr[1], ptr[2]));
        }
        clouds.push_back(cloud);
    }

    return clouds;
}

class Rng
{
public:
    Rng(uint32_t seed) { engine.seed(seed); }

    float next(float low, float high)
    {
        auto t = unif(engine);
        unif(engine); // abandon one sample for identical sequence with numpy
        return (1 - t) * low + t * high;
    }

private:
    std::mt19937 engine;
    std::uniform_real_distribution<float> unif;
};

static constexpr auto RNG_SEED = 1;

std::vector<Eigen::Affine3f> generateRigid()
{
    // Initialize RNG
    Rng rng(RNG_SEED);

    // Generate Euler angles
    std::vector<Eigen::Vector3f> euler(TEST_SIZE);
    for (auto i = 0; i < TEST_SIZE; i++)
        for (auto j = 0; j < 3; j++)
            euler[i][j] = rng.next(-PI, PI);

    // Generate translation components
    std::vector<Eigen::Vector3f> trans(TEST_SIZE);
    for (auto i = 0; i < TEST_SIZE; i++)
        for (auto j = 0; j < 3; j++)
            trans[i][j] = rng.next(-0.5, 0.5);

    // Combine to produce final transformations
    std::vector<Eigen::Affine3f> rigid(TEST_SIZE, Eigen::Affine3f::Identity());
    for (auto i = 0; i < TEST_SIZE; i++)
    {
        auto &angles = euler[i];
        auto R = Eigen::AngleAxisf(angles[2], Eigen::Vector3f::UnitZ()) *
                 Eigen::AngleAxisf(angles[1], Eigen::Vector3f::UnitY()) *
                 Eigen::AngleAxisf(angles[0], Eigen::Vector3f::UnitX());
        rigid[i].prerotate(R);
        rigid[i].translation() << trans[i];
    }

    return rigid;
}

float computeLoss(const Eigen::Affine3f &real, const Eigen::Affine3f &pred)
{
    auto R_loss = (real.rotation() * pred.rotation().transpose() -
                   Eigen::Matrix3f::Identity())
                      .squaredNorm();
    auto t_loss = (real.translation() - pred.translation()).squaredNorm();
    return R_loss + t_loss;
}

struct TestCase
{
    PointCloudPtr src, tgt;
    Eigen::Affine3f real;
};

void testPclReg(pcl::Registration<Point3f, Point3f> &reg,
             const std::vector<TestCase> &cases)
{
    // Initialize metric
    auto loss = 0.f;
    auto time = 0ull;

    // Test each case
    for (auto &cs : cases)
    {
        // Set input clouds
        reg.setInputSource(cs.src);
        reg.setInputTarget(cs.tgt);

        // Align source to target
        Timer timer;
        auto result = boost::make_shared<PointCloud>();
        timer.begin();
        reg.align(*result);
        time += timer.end();
        auto pred = Eigen::Affine3f(reg.getFinalTransformation());

        // Compute metrics for this case
        loss += computeLoss(cs.real, pred);
    }

    // Print metrics
    loss /= cases.size();
    printf("Loss: %f\n", loss);
    time /= cases.size();
    PrintTime(time);
    printf("\n");
}

void testGoICP(GoICP &goicp, const std::vector<TestCase> &cases)
{
    // Initialize metric
    auto loss = 0.f;
    auto time = 0ull;

    // Test each case
    for (auto &cs : cases)
    {
        // Set input clouds
        goicp.SetSource(cs.src);
        goicp.SetTarget(cs.tgt);

        // Align source to target
        Timer timer;
        timer.begin();
        goicp.Register();
        time += timer.end();
        auto pred = Eigen::Affine3f(goicp.GetOptMat());

        // Compute metrics for this case
        loss += computeLoss(cs.real, pred);
        printf(".");
    }
    printf("\n");

    // Print metrics
    loss /= cases.size();
    printf("Loss: %f\n", loss);
    time /= cases.size();
    PrintTime(time);
}

void testAll(std::vector<PointCloudPtr> &src,
             std::vector<Eigen::Affine3f> &trans)
{
    // Generate complete test cases
    std::vector<TestCase> cases;
    for (auto i = 0; i < src.size(); i++)
    {
        auto tgt = boost::make_shared<PointCloud>();
        pcl::transformPointCloud(*src[i], *tgt, trans[i]);
        cases.push_back(TestCase{src[i], tgt, trans[i]});
    }

    // Initialize registration
    // ICP
    pcl::IterativeClosestPoint<Point3f, Point3f> icp;
    icp.setMaximumIterations(10);
    // 4-PCS
    pcl::registration::FPCSInitialAlignment<Point3f, Point3f> fpcs;
    fpcs.setNumberOfSamples(200);
    // Go-ICP
    GoICP goicp(1e-3f, -PI, -PI, -PI, 2 * PI, -.5f, -.5f, -.5f, 1, 0, 2, 300);

    // Test registration methods
    printf("Testing ICP\n");
    testPclReg(icp, cases);
    printf("Testing 4-PCS\n");
    testPclReg(fpcs, cases);
    printf("Testing Go-ICP\n");
    testGoICP(goicp, cases);
}

int main()
{
    auto clouds = loadData("../test.h5");
    auto trans = generateRigid();
    testAll(clouds, trans);
}