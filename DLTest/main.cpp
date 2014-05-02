#include <stdio.h>
#include <iostream>
#include "MNIST.h"
//#include <Windows.h>

#include "SoftmaxRegression.h"

int main()
{
    srand((int)time(NULL));
    mnist::MNIST m("train_images.dat", "train_labels.dat");
    
    boost::optional<mnist::LABLE_DATA_VEC> lbVec = m.GetLableList(5000);
    boost::optional<mnist::IMAGE_DATA_VEC> imVec = m.GetImageList(5000);

    //m.PrintVec(imVec);

    //mnist::LABLE_DATA lable = m.GetLable(98);
    //unsigned char t = *lable;
    //bool r = m.CheckResult(98,3);

    softmax::SoftmaxRegression soft(10, 28*28);


    {
        std::cout << "StudyTimes: " << std::endl;
        soft.Exercise(*lbVec, *imVec);

        mnist::LABLE_DATA_VEC::iterator iterY = lbVec->begin();
        mnist::IMAGE_DATA_VEC::iterator iterX = imVec->begin();
        int len = 28*28;
        double correct = 0;
        double total = lbVec->size();

        std::cout << "Test: " << std::endl;
        for(; (iterY != lbVec->end()) && (iterX != imVec->end()); ++iterY, ++iterX)
        {
            mnist::IMAGE_DATA xData = *iterX;
            softmax::Vector xi(len, 0.0);
            for(int j = 0; j < len; ++j)
                xi[j] = ((xData.get())[j] / 255.0);

            //std::cout << xi << std::endl;
            double res = soft.SoftMaxHx(xi);
            mnist::LABLE_DATA yi = *iterY;

            //printf("T: %d H: %f\n", *yi, res);
            if(softmax::EQ_DOUBLE(*yi, res))
                ++correct;
        }
        std::cout << "Correct: " << correct/total << std::endl;
    }

    return 0;
}