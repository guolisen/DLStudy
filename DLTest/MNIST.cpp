#include <boost/make_shared.hpp>
#include "MNIST.h"

namespace mnist
{

bool MNIST::LoadImageData(unsigned int size)
{
    if(size <= 0)
    {
        std::cout << __FUNCTION__ << " argument error!" << std::endl;
        return false;
    }

    if(!loadImageHead())
    {
        std::cout << "Load file Error!" << std::endl;
        return false;
    }

    if(size > imageHead_.numOfImage)
        size = imageHead_.numOfImage;

    for(unsigned int i = 0; i < size; ++i)
    {
        IMAGE_DATA data = boost::make_shared<unsigned char[]>(imageHead_.rowSize * imageHead_.columnSize);
        imageFin_.read((char*)data.get(), imageHead_.rowSize * imageHead_.columnSize * sizeof(unsigned char));

        imageData_->push_back(data);
    }
    return true;
}

bool MNIST::LoadLableData(unsigned int size)
{
    if(size <= 0)
    {
        std::cout << __FUNCTION__ << " argument error!" << std::endl;
        return false;
    }

    if(!loadLableHead())
    {
        std::cout << "Load file Error!" << std::endl;
        return false;
    }

    if(size > lableHead_.numOfItem)
        size = lableHead_.numOfItem;

    for(unsigned int i = 0; i < size; ++i)
    {
        LABLE_DATA data = boost::make_shared<unsigned char>();
        labelFin_.read((char*)data.get(), sizeof(unsigned char));

        labelData_->push_back(data);
    }

    return true;
}


boost::optional<IMAGE_DATA_VEC>& MNIST::GetImageList(unsigned int size)
{
    if(imageData_ && !imageData_->empty())
        return imageData_;

    if(!LoadImageData(size))
    {
        imageData_ = boost::none;
        return imageData_;
    }
    return imageData_;
}

boost::optional<LABLE_DATA_VEC>& MNIST::GetLableList(unsigned int size)
{
    if(labelData_ && !labelData_->empty())
        return labelData_;

    if(!LoadLableData(size))
    {
        labelData_ = boost::none;
        return labelData_;
    }
    return labelData_;
}

void MNIST::PrintImage(const mnist::IMAGE_DATA& image)
{
    for(int i = 0; i < 28; ++i)
    {
        for(int j = 0; j < 28; ++j)
        {
            if(image[i*28 + j] < 10)
                std::cout << " ";
            else if(image[i*28 + j] < 85)
                std::cout << ".";
            else if(image[i*28 + j] < 170)
                std::cout << "*";
            else
                std::cout << "#";
        }
        std::cout << std::endl;
    }
}

void MNIST::PrintVec(boost::optional<mnist::IMAGE_DATA_VEC>& vec)
{
    mnist::IMAGE_DATA_VEC::iterator iter;
    for(iter = vec->begin(); iter != vec->end(); ++iter)
    {
        PrintImage((*iter));
    }
}

LABLE_DATA MNIST::GetLable(unsigned int position)
{
    LABLE_DATA data = (*labelData_)[position];
    return data;
}

bool MNIST::CheckResult(unsigned int position, unsigned char predict)
{
    if(labelData_)
    {
        LABLE_DATA data = (*labelData_)[position];
        return *data == predict;
    }

    return true;
}

} // mnist