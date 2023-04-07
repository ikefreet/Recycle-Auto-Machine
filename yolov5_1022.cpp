#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include <ctype.h>
#include <fcntl.h>
#include <termios.h>
using namespace std;

// ------------for robot arm--------------------
#include "PCA9685.h"
#include <stdio.h>
#include <thread>         // std::thread
#include <vector>         // std::vector
#include <unistd.h>
#include <string>
#include <string.h>

// -----------------for GPIO --------------------
#include <JetsonGPIO.h>
using namespace GPIO;
#include <chrono>
using namespace chrono;


// -------------------- yolov5 ---------------
#define MIN_PULSE_WIDTH 900
#define MAX_PULSE_WIDTH 2100
#define FREQUENCY 50




int offset = 0;

PCA9685 pwm;

//motor channels
int chan0 = 0;
int chan1 = 1;
int chan2 = 2;
int chan3 = 3;

//motor control deg
int theta0_control = 55;
int theta1_control = 120;
int theta2_control = 90;
int theta3_control = 90;

int theta0_out = 55;
int theta1_out = 120;
int theta2_out = 90;
int theta3_out = 90;

//delay
float delay_time = 5;

//Declaration of Functions used ==================================
int pwmwrite(int& angle, PCA9685 pwm, int& channel);


//def map_pwm(self, x, in_min, in_max, out_min, out_max):
int map_pwm (int x, int in_min, int in_max, int out_min, int out_max) {
        return ((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min);
}

//def _angle_to_analog(self, angle):
int angleToAnalog(int angle) {
      float pulse_wide;
      int analog_value;
      
      pulse_wide = map_pwm(angle,0,180,MIN_PULSE_WIDTH,MAX_PULSE_WIDTH);
      analog_value = int(float(pulse_wide) /  1000000 * FREQUENCY * 4096);
      return (analog_value);
}

int pwmwrite(int& angle, PCA9685 pwm, int& channel) {
    int val = 0;

    if (angle > 180) {
       angle = 179;
    }
    if (angle < 0) {
       angle = 1;
    }
    
    val = angleToAnalog(angle);
    //not sure what offset does
    val += offset;

    //setPWM(self, channel, on, off
    //channel: The channel that should be updated with the new values (0..15)
    //on: The tick (between 0..4095) when the signal should transition from low to high
    //off:the tick (between 0..4095) when the signal should transition from high to low
    
    pwm.setPWM(channel,0,val);
    //usleep(30);
    //cout << "Channel: " << channel << "\tSet to angle: " << angle << "\tVal: " << val << endl;
    return(0);
}

void set_servo(){
	while(1){
		if(theta0_control < theta0_out)
			theta0_control++;
		else if(theta0_control > theta0_out)
			theta0_control--;
		
		if(theta1_control < theta1_out)
			theta1_control++;
		else if(theta1_control > theta1_out)
			theta1_control--;

		if(theta2_control < theta2_out)
			theta2_control++;
		else if(theta2_control > theta2_out)
			theta2_control--;

		if(theta3_control < theta3_out)
			theta3_control++;
		else if(theta3_control > theta3_out)
			theta3_control--;

		pwmwrite(theta0_control, pwm, chan0);
		pwmwrite(theta1_control, pwm, chan1);
		pwmwrite(theta2_control, pwm, chan2);
		pwmwrite(theta3_control, pwm, chan3);

		usleep(delay_time * 1000);
		
	      if(theta0_control == theta0_out and theta1_control == theta1_out and theta2_control == theta2_out and theta3_control == theta3_out)
		        break;
	}
}


// --------------yolov5-------------------------
#define USE_FP32  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4 //0.4
#define CONF_THRESH 0.25	//Confidence, the default value is 0.5, because the effect is not good, modify it to 0.25 and achieve better results
#define BATCH_SIZE 1
 
// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
 
char* my_classes[] = { "Glass", "Can", "Pet" };
 
static int get_width(int x, float gw, int divisor = 8) {
    //return math.ceil(x / divisor) * divisor
    if (int(x * gw) % divisor == 0) {
        return int(x * gw);
    }
    return (int(x * gw / divisor) + 1) * divisor;
}
 
static int get_depth(int x, float gd) {
    if (x == 1) {
        return 1;
    }
    else {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}
 
ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
 
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
 
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
 
    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");
 
    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");
 
    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());
 
    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");
 
    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());
 
    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
 
    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");
 
    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);
 
    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
 
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif
 
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;
 
    // Don't need the network any more
    network->destroy();
 
    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
 
    return engine;
}
 
ICudaEngine* build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
 
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
 
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
 
    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
    auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
    auto spp10 = SPP(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), 3, 5, 7, "model.10");
    auto c3_11 = C3(network, weightMap, *spp10->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.11");
 
    /* ------ yolov5 head ------ */
    auto conv12 = convBlock(network, weightMap, *c3_11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
    auto upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
    ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
    auto cat14 = network->addConcatenation(inputTensors14, 2);
    auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");
 
    auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
    auto upsample17 = network->addResize(*conv16->getOutput(0));
    assert(upsample17);
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
    ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
    auto cat18 = network->addConcatenation(inputTensors18, 2);
    auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");
 
    auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
    auto upsample21 = network->addResize(*conv20->getOutput(0));
    assert(upsample21);
    upsample21->setResizeMode(ResizeMode::kNEAREST);
    upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
    ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors21, 2);
    auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
 
    auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
    ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
    auto cat25 = network->addConcatenation(inputTensors25, 2);
    auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");
 
    auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
    ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
    auto cat28 = network->addConcatenation(inputTensors28, 2);
    auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");
 
    auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
    ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
    auto cat31 = network->addConcatenation(inputTensors31, 2);
    auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");
 
    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
    IConvolutionLayer* det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
    IConvolutionLayer* det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
    IConvolutionLayer* det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);
 
    auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
 
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif
 
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;
 
    // Don't need the network any more
    network->destroy();
 
    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
 
    return engine;
}
 
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
 
    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    assert(engine != nullptr);
 
    // Serialize the engine
    (*modelStream) = engine->serialize();
 
    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}
 
void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}
 
bool parse_args(int argc, char** argv, std::string& engine) {
    if (argc < 3) return false;
    if (std::string(argv[1]) == "-v" && argc == 3) {
        engine = std::string(argv[2]);
    }
    else {
        return false;
    }
    return true;
}


void steering(int ttyfd, int x)
{
    //  차량제어   :  내적값을 이용해서 중심에 가까운걸 구하고, Point inside로 우회전 좌회전 판별
      
      if (x == 0)
            {
                char send[3] = "ll"; //<= 좌회전  아두이노로 송신하는 배열
                printf("입력 : %s\n", send);
                write(ttyfd, send, sizeof(send));
                //usleep(500);
            }
            
      else if (x == 1)
            {
                char send[3] = "rr"; //<= 우회전  아두이노로 송신하는 배열
                printf("입력 : %s\n", send);
                write(ttyfd, send, sizeof(send));
                //usleep(500);

            }
        

      else if (x == 2)
        {
            char send[3] = "gg"; //<= 전진 아두이노로 송신하는 배열:
		printf("입력 : center\n");
            write(ttyfd, send, sizeof(send));
            usleep(500);          
         }
        
      else if (x == 3 || x == 4)
            {
                char send[3] = "tt"; //<= 전진하다 가까워지면 정지 명령  아두이노로 송신하는 배열
                write(ttyfd, send, sizeof(send));
                usleep(500);
            }
}




//int cnt;

 
int main(int argc, char** argv) {
	//--------------- robot arm ---------------------
	//make sure you use the right address values.
	pwm.init(1,0x40);
	usleep(1000);
	cout << "Setting frequency: " << FREQUENCY << endl;
	pwm.setPWMFreq (FREQUENCY);
	usleep(1000);

	pwmwrite(theta0_control, pwm, chan0);
	pwmwrite(theta1_control, pwm, chan1);
	pwmwrite(theta2_control, pwm, chan2);
	pwmwrite(theta3_control, pwm, chan3); 

	// ---------------- GPIO ---------------------
	setmode(BCM);
	setwarnings(false);
	int trig = 17;
	int echo = 18;
	GPIO::setup(17, OUT);
	setup(echo, IN);

	//--------------- yolov5 ---------------------
    cudaSetDevice(DEVICE);
 
    //std::string wts_name = "";
    std::string engine_name = "";
    //float gd = 0.0f, gw = 0.0f;
    //std::string img_dir;
 
    if (!parse_args(argc, argv, engine_name)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -v [.engine] // run inference with camera" << std::endl;
        return -1;
    }
 
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << " read " << engine_name << " error! " << std::endl;
        return -1;
    }
    char* trtModelStream{ nullptr };
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
 
 
    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
 
 
    cv::VideoCapture capture(0, cv::CAP_V4L);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    //cv::VideoCapture capture("../overpass.mp4");
    //int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
    //capture.set(cv::CAP_PROP_FOURCC, fourcc);
    if (!capture.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
 
    int key;
    int fcount = 0;


    //시리얼 통신
	struct termios newtio;
	int ttyfd;
 	char *ttyname = "/dev/ttyACM0";
	ttyfd = open(ttyname, O_RDWR | O_NOCTTY);
	
	if(ttyfd < 0)
	{
		printf( ">> tty Open Fail [%s]\r\n ", ttyname);
		return -1;
	}
	memset( &newtio, 0, sizeof(newtio) );
	
	newtio.c_cflag = B9600 | CS8 | CLOCAL | CREAD | CRTSCTS;
	newtio.c_iflag = IGNPAR;
	newtio.c_oflag = 0;

	//set input mode (non-canonical, no echo,.....)
	newtio.c_lflag     = 0;     // LF recive filter unused
	newtio.c_cc[VTIME] = 0;     // inter charater timer unused
	newtio.c_cc[VMIN]  = 1;     // blocking read until 1 character arrives

	tcflush( ttyfd, TCIFLUSH ); // inital serial port
	tcsetattr( ttyfd, TCSANOW, &newtio ); // setting serial communication
    	printf( "## ttyo1 Opened [%s]\r\n", ttyname);


	int pre = 8;

    int c_glass = 0;
    int c_can = 0;
    int c_pet = 0;
    int c_lnone = 0;

	int cnt = 0;
	int c_left = 0;
	int c_right = 0;
	int c_center = 0;
	int c_none = 0;

    while (1)
    {
        cv::Mat frame;
        capture >> frame;
        if (frame.empty())
        {
            std::cout << "Fail to read image from camera!" << std::endl;
            break;
        }
        fcount++;
        //if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            //cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            cv::Mat img = frame;
            if (img.empty()) continue;
            cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                for (int col = 0; col < INPUT_W; ++col) {
                    data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }
 
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        int fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }


        std::vector<int> b_c; 
        std::vector<int> b_label;
        std::vector<int> b_state;



        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];

		if(res.size() == 0){
                b_state.push_back(4);
                b_label.push_back(3);
		}
            //std::cout << res.size() << std::endl;
            //cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = get_rect(frame, res[j].bbox);
                cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                std::string label = my_classes[(int)res[j].class_id];
                cv::putText(frame, label, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                std::string jetson_fps = "Jetson Nano FPS: " + std::to_string(fps);
                cv::putText(frame, jetson_fps, cv::Point(11, 80), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::Point ptTL = r.tl();
                cv::Point ptBr = r.br();
                cv::Point ptcenter = (ptTL + ptBr) / 2;
                cv::Point xycenter(640, 360);

                int d = ptcenter.dot(xycenter);
                int dd = abs(d - 539200); 
                b_c.push_back(dd);

                cv::Rect left(0, 0, 560, 720);
                cv::Rect right(720, 0, 560, 720);
                cv::Rect center(561, 0, 159, 720);
                cv::Rect nready(0 ,0 , 1280, 560);
		cv::Rect ready(0, 600, 1280, 120);
                int state;

                bool bleft = left.contains(ptcenter);
                bool bright = right.contains(ptcenter);
                bool bcenter = center.contains(ptcenter);
                bool bnready = nready.contains(ptBr);
		        bool bready = ready.contains(ptBr);



                if(bleft) {state = 0;}         // 0일경우 left회전
                else if(bright) {state = 1;}   // 1일경우 right회전
                else if(bcenter) {state = 2;} // 2일경우 전진
         

                b_state.push_back(state);

                int ilabel;
                if(label == "Glass" ) { ilabel = 0;}
                else if(label ==  "Can" ) { ilabel = 1;}
                else if(label == "Pet" ) { ilabel = 2;}     
         
                b_label.push_back(ilabel);

            }
            //cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
        }



        int m = b_c.size();

	  if(m > 1) {
	     int temp = 0;
   	     for (int b = 0; b < m; b++) {
   	         for (int j =0; j < m-1; j++){
   	             if(b_c[j] > b_c[j+1]){
   	                temp = b_c[j];
   	                b_c[j] = b_c[j+1];
   	                b_c[j+1] = temp;

   	                temp = b_state[j];
   	                b_state[j] = b_state[j+1];
   	                b_state[j+1] = temp;
      
   	                temp = b_label[j];
   	                b_label[j] = b_label[j+1];
   	                b_label[j+1] = temp;
    	            }
    	        }
    	    }    
   	 }

     		
        int D_label; // = b_label[0];
	/*
	  int DP_label = b_label[0];
	  int D_label = 3;
	   for (int i = 3; i >=0; i--)
	    {
		cl[i+1] = cl[i];
	    }
	    cl[0] = DP_label;

	  if (cl[0] == cl[1] && cl[1] == cl[2] && cl[2] == cl[3] && cl[3] == cl[4])
	{
		D_label = cl[0];
	}
	*/

        int D_state;// = b_state[0];

    if(b_label[0] == 0) c_glass++;
	else if(b_label[0] == 1) c_can++;
	else if(b_label[0] == 2) c_pet++;
	else if(b_label[0] == 3) c_lnone++;    

	if(b_state[0] == 0) c_left++;
	else if(b_state[0] == 1) c_right++;
	else if(b_state[0] == 2) c_center++;
	else if(b_state[0] == 4) c_none++;

	cnt++; 

	if (cnt >= 5){
		int c_max = 0;
		
		if(c_max < c_left){
			c_max = c_left;
			D_state = 0;
		}
		if(c_max < c_right){
			c_max = c_right;
			D_state = 1;
		}
		if(c_max < c_center){
			c_max = c_center;
			D_state = 2;
		}
		if(c_max < c_none){
			c_max = c_none;
			D_state = 3;
		}

        int c_lmax = 0;

        if(c_lmax < c_glass){
			c_lmax = c_glass;
			D_label = 0;
		}
		if(c_lmax < c_can){
			c_lmax = c_can;
			D_label = 1;
		}
		if(c_lmax < c_pet){
			c_lmax = c_pet;
			D_label = 2;
		}
		if(c_lmax < c_lnone){
			c_lmax = c_lnone;
			D_label = 3;
		}

		c_left = 0;
		c_right = 0;
		c_center = 0;
		c_none = 0;
        c_glass = 0;
        c_can = 0;
        c_pet = 0;
        c_lnone = 0;
		cnt =0;
	}	   


       // std::string t_label = "empty";
       // std::string t_state = "empty";

/*        if(D_label == 0) { t_label = "Glass"; }
        else if(D_label == 1) { t_label = "Can"; }
        else if(D_label == 2) { t_label = "Pet"; }
*/
 /*       if(D_state == 0) { t_state = "Left"; }
        else if(D_state == 1) { t_state = "Right"; }
        else if(D_state == 2) { t_state = "Forward"; }
        else if(D_state == 3) { t_state = "Ready"; }
*/
//        cv::putText(frame, t_label, cv::Point(640, 80), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        //cv::putText(frame, t_state, cv::Point(640, 200), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

	//printf("%d", D_state);
	if (pre != D_state)
		{
			steering(ttyfd, D_state);
		}
		//delay(10);
	  

        cv::imshow("yolov5", frame);
        fcount = 0;
		
	 
        key = cv::waitKey(1);

      if (key == 'q') {
            break;
        }

	// for ultra wave
	system_clock::time_point start_time, end_time;
	float distance;

	GPIO::output(trig, 0);
	GPIO::output(trig, 1);
	usleep(10);
	GPIO::output(trig, 0);
	

	while(GPIO::input(echo) == 0);
	start_time = system_clock::now();
	
	while(GPIO::input(echo) == 1);
	end_time = system_clock::now();
	
	microseconds micro = duration_cast<microseconds>(end_time - start_time);

	distance = (float)micro.count() / 29. /2.;

	

	if(distance < 28){
		steering(ttyfd, 3);
		printf("detected, distance %.2f cm\n", distance);
	
		if(D_label == 1){
			theta0_out = 55;
			theta1_out = 120;
			theta2_out = 170;
			theta3_out = 130; 
			set_servo();

			theta0_out = 35;
			theta1_out = 55;
			theta2_out = 170;
			theta3_out = 130;   
			set_servo();

			theta0_out = 65;
			theta1_out = 25;
			theta2_out = 170;
			theta3_out = 130;   
			set_servo();

			theta0_out = 55;
			theta1_out = 120;
			theta2_out = 170;
			theta3_out = 130;   
			set_servo();
			
			theta0_out = 55;
			theta1_out = 120;
			theta2_out = 90;
			theta3_out = 90;   
			set_servo();
		}
		if(D_label == 2){
			theta0_out = 55;
			theta1_out = 120;
			theta2_out = 50;
			theta3_out = 10; 
			set_servo();

			theta0_out = 35;
			theta1_out = 55;
			theta2_out = 50;
			theta3_out = 10;   
			set_servo();

			theta0_out = 65;
			theta1_out = 25;
			theta2_out = 50;
			theta3_out = 10;   
			set_servo();

			theta0_out = 55;
			theta1_out = 120;
			theta2_out = 50;
			theta3_out = 10;   
			set_servo();
			
			theta0_out = 55;
			theta1_out = 120;
			theta2_out = 90;
			theta3_out = 90;   
			set_servo();
		}
		if(D_label == 0){
			theta0_out = 55;
			theta1_out = 120;
			theta2_out = 50;
			theta3_out = 130; 
			set_servo();

			theta0_out = 35;
			theta1_out = 55;
			theta2_out = 50;
			theta3_out = 130;   
			set_servo();

			theta0_out = 65;
			theta1_out = 25;
			theta2_out = 50;
			theta3_out = 130;   
			set_servo();

			theta0_out = 55;
			theta1_out = 120;
			theta2_out = 50;
			theta3_out = 130;   
			set_servo();
			
			theta0_out = 55;
			theta1_out = 120;
			theta2_out = 90;
			theta3_out = 90;   
			set_servo();
		}
	}

	else if (key == 'a'){
		 D_state = 0;
		steering(ttyfd, D_state);
	}

	else if (key == 'd'){
		 D_state = 1;
		steering(ttyfd, D_state);
	}

	else if (key == 'w'){
		 D_state = 2;
		steering(ttyfd, D_state);
	}
	
	else if (key == 's'){
		 D_state = 3;
		steering(ttyfd, D_state);
	}

	pre = D_state;

	
    }
 
    capture.release();
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
 
    return 0;
}
