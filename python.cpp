#include <boost/python.hpp>
#include "scene_flow_impair.h"

#ifdef NDEBUG
#   define MODULE PySceneFlow
#else
#   define MODULE PySceneFlowD
#endif

cv::Mat getDisplacement(const PD_flow_opencv &flow)
{
    cv::Size sz(flow.rows, flow.cols);
    cv::Mat
        x = cv::Mat(sz, cv::DataType<float>::type, flow.dxp),
        y = cv::Mat(sz, cv::DataType<float>::type, flow.dyp),
        z = cv::Mat(sz, cv::DataType<float>::type, flow.dzp);
    cv::Mat ret;
    std::vector<cv::Mat> xyz{ x, y, z };
    cv::merge(xyz, ret);
    cv::Mat tret;
    cv::transpose(ret, tret);
    return tret;
}

BOOST_PYTHON_MODULE(MODULE)
{
    using namespace boost::python;

    class_<PD_flow_opencv>("PD_flow_opencv", init<unsigned int>(args("rows_config")))
        .def("loadRGBDFrames", static_cast<bool(PD_flow_opencv::*)(const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)>(&PD_flow_opencv::loadRGBDFrames),
        (arg("i1"), arg("d1"), arg("i2"), arg("d2")))
        .def("solveSceneFlowGPU", &PD_flow_opencv::solveSceneFlowGPU)
        .def("initializeCUDA", static_cast<void(PD_flow_opencv::*)(unsigned int, unsigned int)>(&PD_flow_opencv::initializeCUDA),
        (arg("rows"), arg("cols")))
        .def("freeGPUMemory", &PD_flow_opencv::freeGPUMemory)
        .def("getField", getDisplacement)
        ;
}