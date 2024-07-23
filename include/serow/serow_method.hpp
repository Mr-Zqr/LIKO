#include "ContactDetection.h"
#include <Eigen/Dense>

class SerowMethod : ContactDetection
{
public:
    SerowMethod(bool useGEM = false)
    :lfoot_frame("l_ankle"),rfoot_frame("r_ankle"),LosingContact(80),
    foot_polygon_xmin(-0.1),foot_polygon_xmax(0.1),foot_polygon_ymin(-0.05),foot_polygon_ymax(0.05),
    lforce_sigma(3.0),rforce_sigma(3.0),lcop_sigma(0.005),rcop_sigma(0.005),VelocityThres(0.2),lvnorm_sigma(0.05),rvnorm_sigma(0.05),
    ContactDetectionWithCOP(true),ContactDetectionWithKinematics(true),probabilisticContactThreshold(0.95),
    medianWindow(5),LegHighThres(250),LegLowThres(125),StrikingContact(1200),firstContact(true),
    LLegForceFilt(Eigen::Vector3d::Zero()),RLegForceFilt(Eigen::Vector3d::Zero()),copl(Eigen::Vector3d::Zero()),copr(Eigen::Vector3d::Zero()),g(9.81)
    {
        lmdf = MediatorNew(medianWindow);
        rmdf = MediatorNew(medianWindow);
    };

    ~SerowMethod()
    {}

    void LLeg_FT(Eigen::Vector3d LLegGRF, Eigen::Vector3d LLegGRT)
    {
        LLegForceFilt = LLegGRF;
        MediatorInsert(lmdf, LLegGRF(2)); // 大概是某种窗口数量固定的中位数滤波
        LLegForceFilt(2) = MediatorMedian(lmdf);  // 同上

        weightl = 0; // double
        copl = Eigen::Vector3d::Zero(); // Vector3d //左脚CoP
        if (LLegGRF(2) >= LosingContact)  //阈值判断接触，或者应该说是过滤白噪声（？）
        {
            copl(0) = -LLegGRT(1) / LLegGRF(2);
            copl(1) = LLegGRT(0) / LLegGRF(2);
            weightl = LLegGRF(2) / g; //左脚质量
        }
        else
        {
            copl = Eigen::Vector3d::Zero();
            LLegGRF = Eigen::Vector3d::Zero();
            LLegGRT = Eigen::Vector3d::Zero();
            weightl = 0.0;
        }
    }

    void RLeg_FT(Eigen::Vector3d RLegGRF, Eigen::Vector3d RLegGRT)
    {
        RLegForceFilt = RLegGRF;
        MediatorInsert(rmdf, RLegGRF(2));
        RLegForceFilt(2) = MediatorMedian(rmdf);

        weithtr = 0;
        copr = Eigen::Vector3d::Zero();
        if (RLegGRF(2) >= LosingContact)
        {
            copr(0) = -RLegGRT(1) / RLegGRF(2);
            copr(1) = RLegGRT(0) / RLegGRF(2);
            weithtr = RLegGRF(2) / g;
        }
        else
        {
            copr = Eigen::Vector3d::Zero();
            RLegGRF = Eigen::Vector3d::Zero();
            RLegGRT = Eigen::Vector3d::Zero();
            weithtr = 0.0;
        }
    }

    std::string computeKinTFs()
    {
        if (firstContact)
        {
            cd = new ContactDetection();
            if (useGEM)
            {
                cd->init(lfoot_frame, rfoot_frame, LosingContact, LosingContact, foot_polygon_xmin, foot_polygon_xmax,
                        foot_polygon_ymin, foot_polygon_ymax, lforce_sigma, rforce_sigma, lcop_sigma, rcop_sigma, VelocityThres,
                        lvnorm_sigma, rvnorm_sigma, ContactDetectionWithCOP, ContactDetectionWithKinematics, probabilisticContactThreshold, medianWindow);
            }
            else
            {
                cd->init(lfoot_frame, rfoot_frame, LegHighThres, LegLowThres, StrikingContact, VelocityThres, medianWindow);
            }

            firstContact = false;
        }
        if (useGEM)
        {
            // cd->computeSupportFoot(LLegForceFilt(2), RLegForceFilt(2),
            //                     copl(0), copl(1), copr(0), copr(1),
            //                     vwl.norm(), vwr.norm());
        }
        else
        {
            cd->computeForceWeights(LLegForceFilt(2), RLegForceFilt(2));
            cd->SchmittTrigger(LLegForceFilt(2), RLegForceFilt(2));
        }
        return cd->getSupportLeg();
    }

private:
    std::string lfoot_frame, rfoot_frame;
    double LosingContact, foot_polygon_xmin, foot_polygon_xmax, foot_polygon_ymin, foot_polygon_ymax;
    double lforce_sigma, rforce_sigma, lcop_sigma, rcop_sigma;
    double VelocityThres, lvnorm_sigma, rvnorm_sigma;
    bool ContactDetectionWithCOP, ContactDetectionWithKinematics;
    double probabilisticContactThreshold;
    int medianWindow; 
    double LegHighThres, LegLowThres, StrikingContact;
    bool firstContact, useGEM;
    Eigen::Vector3d LLegForceFilt, RLegForceFilt;
    Mediator *lmdf, *rmdf;
    double weightl, weithtr;
    Eigen::Vector3d copl, copr;
    double g;
    ContactDetection *cd;
};