#pragma once

#include "LinearDT.hpp"

#include <ctime>
#include <queue>
#include <pcl/registration/icp.h>

#define SQRT3 1.732050808

struct RotationNode
{
    float a, b, c, w;
    float ub, lb;
    int l;
    friend bool operator<(const RotationNode &n1, const RotationNode &n2)
    {
        if (n1.lb != n2.lb)
            return n1.lb > n2.lb;
        else
            return n1.w < n2.w;
    }
};

struct TranslationNode
{
    float x, y, z, w;
    float ub, lb;
    friend bool operator<(const TranslationNode &n1, const TranslationNode &n2)
    {
        if (n1.lb != n2.lb)
            return n1.lb > n2.lb;
        else
            return n1.w < n2.w;
    }
};

#define MAXROTLEVEL 20

class GoICP
{
public:
    GoICP(float MSE_Thresh, float rotMinX, float rotMinY, float rotMinZ,
          float rotWidth, float transMinX, float transMinY, float transMinZ, 
          float transWidth, float trim_Fraction, float expand_factor, 
          uint32_t div);
    float Register();
    void SetSource(PointCloudPtr src);
    void SetTarget(PointCloudPtr tgt);
    const Matrix4f GetOptMat();
    const int GetModelSize();
    const int GetDataSize();

private:
    PointCloudPtr pModel, pData;

    // Thresholds for termination
    float mseThresh;
    float sseThresh;
    float icpThresh;

    // Optimal parameters (dynamically change during runtime)
    float optError;
    Matrix4f optMat;
    RotationNode optNodeRot;
    TranslationNode optNodeTrans;

    // Trimming related parameters
    float trimFraction;
    int inlierNum;
    bool doTrim;

    // Transformation nodes
    RotationNode initNodeRot;
    TranslationNode initNodeTrans;

    // 3D Euclidean Distance Transform used to speedup SSE calculation
    std::unique_ptr<LinearDT> ldt = nullptr;
    float expandFactor;
    uint32_t div;

    // Temporary variables
    std::vector<float> minDis;
    float **maxRotDis;

    // Standard ICP implementation
    pcl::IterativeClosestPoint<Point3f, Point3f> icp;

    float runICP(Matrix4f &trans_now); //parameter is current rot & trans matrix
    float innerBnB(float *maxRotDisL, TranslationNode *nodeTransOut,
                   const std::vector<Point3f> &pDataTemp);
    float outerBnB();
    void initialize();
    void clear();
};
