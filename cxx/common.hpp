#pragma once

// Standard library
#include <cstdint>
#include <cmath>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

// Boost
#include <boost/make_shared.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// PCL
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

// TBB
#include <tbb/tbb.h>

// Type alias
using IndexT = uint32_t;
#define INVALID_INDEX IndexT(-1)
using Point3f = pcl::PointXYZ;
using Vector3f = Eigen::Vector3f;
using Vector3i = Eigen::Vector3i;
using Matrix3f = Eigen::Matrix3f;
using Matrix4f = Eigen::Matrix4f;
using PointCloud = pcl::PointCloud<Point3f>;
using PointCloudPtr = boost::shared_ptr<PointCloud>;

// Mathematical constants
#define PI 3.1415926536

// Overloaded point and vector operators
#define SQ(x) ((x) * (x))

inline Vector3f operator-(const Point3f &p1, const Point3f &p2)
{
    return {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
}

inline Point3f operator+(const Point3f &p, const Vector3f &v)
{
    return {p.x + v.x(), p.y + v.y(), p.z + v.z()};
}

inline Point3f operator-(const Point3f &p, const Vector3f &v)
{
    return {p.x - v.x(), p.y - v.y(), p.z - v.z()};
}

inline float DistSq(const Point3f &p1, const Point3f &p2)
{
    return (p1 - p2).squaredNorm();
}

inline float Dist(const Point3f &p1, const Point3f &p2)
{
    return (p1 - p2).norm();
}

inline Vector3f ToVector3f(const Point3f &p)
{
    return {p.x, p.y, p.z};
}

inline Vector3f ToPoint3f(const Vector3f &v)
{
    return {v.x(), v.y(), v.z()};
}

inline int MaxDim(const Vector3f &v)
{
    auto dim = 0u;
    auto val = -INFINITY;
    for (auto i = 0u; i < 3; i++)
        if (v[i] > val)
        {
            dim = i;
            val = v[i];
        }
    return dim;
}

inline float MaxComp(const Vector3f &v)
{
    return std::max(std::max(v.x(), v.y()), v.z());
}

struct Bound3f
{
    Point3f min, max;

    Bound3f(const Point3f &min, const Point3f &max) : min(min), max(max)
    {
        for (auto i = 0u; i < 3; i++)
            if (this->min.data[i] > this->max.data[i])
                std::swap(this->min.data[i], this->max.data[i]);
    }

    Bound3f(const PointCloud &cloud)
    {
        pcl::getMinMax3D(cloud, min, max);
    }

    Vector3f Diagonal() const
    {
        return max - min;
    }
};

inline std::ostream &operator<<(std::ostream &os, const Bound3f &bnd)
{
    os << "min: " << bnd.min << " max: " << bnd.max << '\n';
    return os;
}