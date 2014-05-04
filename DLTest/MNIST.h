#ifndef MNIST_HEAD_FILE_H
#define MNIST_HEAD_FILE_H

#include <string>
#include <vector>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/optional.hpp>

#define Swap16(s) ((((s) & 0xff) << 8) | (((s) >> 8) & 0xff))

#define Swap32(l) (((l) >> 24) | \
           (((l) & 0x00ff0000) >> 8)  | \
           (((l) & 0x0000ff00) << 8)  | \
           ((l) << 24))

namespace mnist
{



struct LABLE_HEAD_STRU
{
    unsigned int magicNum;
    unsigned int numOfItem;
};

struct IMAGE_HEAD_STRU
{
    unsigned int magicNum;
    unsigned int numOfImage;
    unsigned int rowSize;
    unsigned int columnSize;
};

typedef boost::shared_ptr<unsigned char[]> IMAGE_DATA;
typedef std::vector<IMAGE_DATA> IMAGE_DATA_VEC;

typedef boost::shared_ptr<unsigned char> LABLE_DATA;
typedef std::vector<LABLE_DATA> LABLE_DATA_VEC;

class MNIST
{
public:
    MNIST(const std::string& imageFileName, const std::string& lableFileName): 
        imageFileName_(imageFileName), lableFileName_(lableFileName),
        imageData_(std::vector<IMAGE_DATA>(0)), labelData_(std::vector<LABLE_DATA>(0)){}
    virtual ~MNIST()
    {
        if(imageFin_)
            imageFin_.close();
        if(labelFin_)
            labelFin_.close();
    }

    virtual boost::optional<IMAGE_DATA_VEC> GetImageList(unsigned int fromImage, unsigned int toImage);
    virtual boost::optional<LABLE_DATA_VEC> GetLableList(unsigned int fromImage, unsigned int toImage);
    virtual void PrintVec(boost::optional<mnist::IMAGE_DATA_VEC>& vec);
    virtual void PrintVec()
    {
        PrintVec(imageData_);
    }
    virtual void PrintImage(const mnist::IMAGE_DATA& image);
    virtual LABLE_DATA GetLable(unsigned int position);
    virtual bool CheckResult(unsigned int position, unsigned char predict);

private:
    virtual bool LoadImageData(unsigned int size);
    virtual bool LoadLableData(unsigned int size);

    virtual void swapHead(LABLE_HEAD_STRU& head)
    {
        head.magicNum  = Swap32(head.magicNum);
        head.numOfItem = Swap32(head.numOfItem);
    }
    virtual void swapHead(IMAGE_HEAD_STRU& head)
    {
        head.magicNum   = Swap32(head.magicNum);;
        head.numOfImage = Swap32(head.numOfImage);;
        head.rowSize    = Swap32(head.rowSize);;
        head.columnSize = Swap32(head.columnSize);;
    }

    template<typename HeadType, unsigned int magicNum>
    bool loadFileHead_templ(const std::string& fileName, HeadType& headData, std::ifstream& dataFin)
    {
        dataFin.open(fileName.c_str(), std::ios::binary);
        if(!dataFin.is_open())
        {
            std::cout << "Can not find the file of " << fileName << std::endl;
            return false;
        }

        dataFin.read((char*)&headData, sizeof(headData));
        swapHead(headData);
        if(headData.magicNum != magicNum)
        {
            std::cout << "It is not the taget file(" << fileName << ")" << std::endl;
            return false;
        }

        return true;
    }

    virtual bool loadImageHead()
    {
        return loadFileHead_templ<IMAGE_HEAD_STRU, 0x00000803>(imageFileName_, imageHead_, imageFin_);
    }
    virtual bool loadLableHead()
    {
        return loadFileHead_templ<LABLE_HEAD_STRU, 0x00000801>(lableFileName_, lableHead_, labelFin_);
    }


private:
    std::string imageFileName_;
    std::string lableFileName_;

    std::ifstream imageFin_;
    std::ifstream labelFin_;

    LABLE_HEAD_STRU lableHead_;
    IMAGE_HEAD_STRU imageHead_;

    boost::optional<IMAGE_DATA_VEC> imageData_;
    boost::optional<LABLE_DATA_VEC> labelData_;
};



} // mnist

#endif