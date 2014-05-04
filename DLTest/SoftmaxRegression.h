#ifndef SOFTMAX_REGRESSION_H
#define SOFTMAX_REGRESSION_H

#include <math.h>
#include <vector>

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/shared_ptr.hpp>


namespace softmax
{

inline bool EQ_DOUBLE(double left, double right)
{
    const double EPS = 1e-5;
    return (fabs(left - right) < EPS)? true: false;
}

typedef mtl::dense_vector<double> Vector;
class SoftmaxRegression
{
public:
    SoftmaxRegression(int kinds, double inputNum): theta_(inputNum, kinds), bias_(kinds), kinds_(kinds) 
    {
        for(int i = 0; i < kinds; ++i)
        {
            bias_[i] = 0;
            for(int j = 0; j < inputNum; ++j)
                theta_.vector(i)[j] = 0;
        }
    }
public:
    template<typename T, typename T2>
    bool Exercise(std::vector<boost::shared_ptr<T> >& y, std::vector<boost::shared_ptr<T2> >& x)
    {
        for(int i = 0; i < kinds_; ++i)
        {
            //std::cout << "theta: " << i << std::endl;
            theta_.vector(i) += 0.0013 * Gradient(y, x, i);
            //std::cout << theta_.vector(i) << std::endl;
        }

        return true;
    }

    double SoftMaxHx(const Vector& x)
    {
        double maxP = 0.0;
        double result  = 0;
        for(int i = 0; i < kinds_; ++i)
        {
            double temp = Pn(i, x);
            if(temp > maxP)
            {
                maxP = temp;
                result = i;
            }
        }
        return (double)result;
    }
protected:
    double Pn(unsigned int i, const Vector& xi)
    {
        unsigned int thetaNum = num_cols(theta_);
        unsigned int thetaRows = num_rows(theta_);
        if(i > thetaNum)
            return 0.0;
        //std::cout << theta_.vector(i) << std::endl;
        //std::cout << xi << std::endl;
        double temp = trans(theta_.vector(i)) * xi;
        double counter = exp(temp);

        double denominator = 0.0;
        for(int count = 0; count < thetaNum; ++count)
        {
            double tempCounter = trans(theta_.vector(count)) * xi;
            double counterTemp = exp(tempCounter);

            denominator += counterTemp;
        }

        return counter/denominator;
    }
    
    Vector GradientDelta(double j, double y, Vector& x, double Pn)
    {
        Vector delta;
        if(EQ_DOUBLE(y, j))
            delta = x * (1 - Pn);
        else
            delta = x * Pn;

        return delta;
    };

    template<typename T, typename T2>
    Vector Gradient(std::vector<boost::shared_ptr<T> >& y, std::vector<boost::shared_ptr<T2> >& x, int thetaN)
    {
        std::vector<boost::shared_ptr<T> >::iterator iterY = y.begin();
        std::vector<boost::shared_ptr<T2> >::iterator iterX = x.begin();
        int len = 28*28;
        Vector gradientVec(len, 0.0);
        int m = y.size();
        for(; (iterY != y.end()) && (iterX != x.end()); ++iterY, ++iterX)
        {
            boost::shared_ptr<T2> xData = *iterX;
            Vector xi(len, 0.0);
            for(int j = 0; j < len; ++j)
                xi[j] = (xData.get())[j] / 255.0;

            T yi = **iterY;

            //std::cout << "xi: " << xi << std::endl;
            //printf("yi: %d\n", yi);

            Vector tmp = GradientDelta(thetaN, (double)yi, xi, Pn(thetaN, xi));
            //std::cout << "tmp vector: " << tmp << std::endl;
            gradientVec += (tmp / m);
            //std::cout << "GradientVec vector: " << gradientVec << std::endl;
        }

        return gradientVec;
    };

protected:
    mtl::multi_vector<Vector> theta_;
    Vector bias_;
    int kinds_;
};


};


#endif