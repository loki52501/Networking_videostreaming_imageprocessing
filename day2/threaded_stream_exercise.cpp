#include <iostream>
#include <thread>
#include <deque>
#include <condition_variable>
#include <mutex>
#include <algorithm>
#include <functional>
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace std;
using namespace cv;
using namespace chrono;

size_t kQueueCapacity=6;
milliseconds slodelay{500};

struct FrameItem{
    Mat frame;
    steady_clock::time_point timestamp;
};

struct sharedstate{
    deque<FrameItem> queue;
    condition_variable cv;
    mutex locker;
    bool producerDone=false;
    bool injectDelay=false;
    size_t framesprocessed=0;
    size_t bufferunderruns=0;
    size_t injectDelays=0;
    size_t maxdepth = 0;
    steady_clock::duration totaldelay{0};
};

void renderFrame(const Mat& Frame, sharedstate& state)
{
    Mat gray;
    cvtColor(Frame, gray, COLOR_BGR2GRAY);
     const auto avgdelay=state.totaldelay/state.framesprocessed;
    
     ostringstream os;
     os<<" frame per second =="<<duration_cast<milliseconds>(avgdelay).count()<<" fps";
     setWindowTitle("heya",os.str());
    imshow("heya", gray);
    if(waitKey(1)==27)
    return;

}
void streamer(sharedstate& state, milliseconds frameInterval)
{
    while (true)
    {
        FrameItem item;
        {
            unique_lock<mutex> locker(state.locker);
            state.cv.wait(locker, [&]{ return !state.queue.empty() || state.producerDone; });
            if (state.queue.empty() && state.producerDone)
                break;
            item = move(state.queue.front());
            state.queue.pop_front();
            state.cv.notify_all();
        }

        const auto now = steady_clock::now();
        const auto delay = now - item.timestamp;
        state.totaldelay += delay;
        state.framesprocessed++;
        if (delay > frameInterval * 2) {
            ++state.bufferunderruns;
            cout << "[WARN] Buffer underrun detected\n";
        }
        renderFrame(item.frame,state);
    }
}

void dataer(VideoCapture& cam, sharedstate& state, milliseconds frameinterval)
{int framecount=0;
    try{
        while(true){
        FrameItem item;
if(!cam.read(item.frame))
{
    lock_guard<mutex> locks(state.locker);
    state.producerDone=true;
    state.cv.notify_all();
    break;
}
item.timestamp=steady_clock::now();

{
    unique_lock<mutex> lockers(state.locker);
    state.cv.wait(lockers,[&]{return state.queue.size()<kQueueCapacity;});
    state.queue.push_back(move(item));
    state.maxdepth = max(state.maxdepth, state.queue.size());
}
state.cv.notify_all();
framecount++;
duration<double> dur=steady_clock::now()-item.timestamp;
if(state.injectDelay && dur.count()>=1 )
{state.injectDelays++;
this_thread::sleep_for(slodelay);

    }
else
{
    this_thread::sleep_for(frameinterval);
}
}
}
    catch(exception&ex)
    {
        lock_guard<mutex>lock(state.locker);
        state.producerDone=true;
        state.cv.notify_all();
        cerr<<" data taker made an error:  "<<ex.what()<<"\n";
    }
}

int main()
{ 
  VideoCapture cam(0);
if(!cam.isOpened())
    {
        cerr<<" webcam is not there or not working properly;";
        return -1;
    }
cam.set(CAP_PROP_FRAME_WIDTH,1280);
cam.set(CAP_PROP_FRAME_HEIGHT,720); // setting the cam size.
sharedstate state;
const double targetfps=30.0;
auto frameinterval=milliseconds(static_cast<int>(1000.0/targetfps));

thread dat(dataer,ref(cam),ref(state),frameinterval);
thread stre(streamer,ref(state),frameinterval);
if(dat.joinable())
dat.join();
if(stre.joinable())
stre.join();

if(state.framesprocessed>0){
 const auto avgdelay=state.totaldelay/state.framesprocessed;
 cout<<"\n=== Stream Summary ===\n";
 cout<<"Frames Processed: "<<state.framesprocessed<<"\n";
 auto avgdelayMs = duration_cast<milliseconds>(avgdelay).count();
 cout<<"Average delays: "<<avgdelayMs<<" ms\n";
 cout<<"Max queue depth: "<<state.maxdepth<<"\n";
 cout<<"Buffer Underruns: "<<state.bufferunderruns<<"\n";
 cout<<"Injected delays: "<<state.injectDelays<<"\n";
}
else{
    cout<<"no frames were captured.\n";
}


    return 0;
}
