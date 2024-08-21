std::vector<TextBox> getTextBoxes(ncnn::Net & net, const cv::Mat & src, float boxScoreThresh, float boxThresh, float unClipRatio)
{
    int width = src.cols;
    int height = src.rows;
    int min_size = 32;
    int max_size = 640;
    bool is_pad_mode = false;
    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w>max_size || h>max_size) {
        if (w > h)
        {
            scale = (float)max_size / w;
            w = max_size;
            h = h * scale;
        }
        else
        {
            scale = (float)max_size / h;
            h = max_size;
            w = w * scale;
        }
    } else if (w<min_size || h<min_size) {
        if (w < h)
        {
            scale = (float)min_size / w;
            w = min_size;
            h = h * scale;
        }
        else
        {
            scale = (float)min_size / h;
            h = min_size;
            w = w * scale;
        }
    }
    
    ncnn::Mat in_pad;
    int wpad = 0;
    int hpad = 0;
    if (is_pad_mode) {
        ncnn::Mat input = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);
        wpad = (w + 31) / 32 * 32 - w;
        hpad = (h + 31) / 32 * 32 - h;
        ncnn::copy_make_border(input, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    } else {
        // using the resize mode instead of padding mode
        h = roundInt((float)h / 32) * 32;
        w = roundInt((float)w / 32) * 32;
        in_pad = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);
    }
    
    const float meanValues[3] = { 0.485 * 255, 0.456 * 255, 0.406 * 255 };
    const float normValues[3] = { 1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0 };

    in_pad.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = net.create_extractor();

    extractor.input("input", in_pad);
    ncnn::Mat out;
    extractor.extract("output", out);

    cv::Mat fMapMat(in_pad.h, in_pad.w, CV_32FC1, (float*)out.data);
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    cv::dilate(norfMapMat, norfMapMat, cv::Mat(), cv::Point(-1, -1), 1);

    std::vector<TextBox> result = findRsBoxes(fMapMat, norfMapMat, boxScoreThresh, 2.0f);
    for(int i = 0; i < result.size(); i++)
    {
        for(int j = 0; j < result[i].boxPoint.size(); j++)
        {
            float x, y;
            if (is_pad_mode) {
                x = ((float)result[i].boxPoint[j].x-(wpad/2))/scale;
                y = ((float)result[i].boxPoint[j].y-(hpad/2))/scale;
            } else {
                x = ((float)result[i].boxPoint[j].x/w*width-(wpad/2))/scale;
                y = ((float)result[i].boxPoint[j].y/h*height-(hpad/2))/scale;
            }
            x = std::max(std::min(x,(float)(width-1)),0.f);
            y = std::max(std::min(y,(float)(height-1)),0.f);
            result[i].boxPoint[j].x = x;
            result[i].boxPoint[j].y = y;
        }
    }

    return result;
}
