#include <opencv2/opencv.hpp>
#include <iostream>
#include<chrono>
#include <optional>
using namespace std;
using namespace cv;
using namespace chrono;
mutex mtx;
condition_variable frames_ready;
bool producer_done=false;

struct frame{
    frame(Mat val):id  (next_id()), piece(val), s(get_time()) {}
    int id,framecounter=0;
    steady_clock::time_point s;
    Mat piece;
private:
int next_id()
{
    static atomic<int>counter(0);
    return counter.fetch_add(1,memory_order_relaxed);
}
steady_clock::time_point get_time()
{
    return high_resolution_clock::now();
}
};

void stream(vector<frame>& frames)
{
            while(true)
            {
                optional<frame> work;
                {
                    unique_lock<mutex> lockee(mtx);
                    frames_ready.wait(lockee,[&]{return !frames.empty()|| producer_done;});
                    if(frames.empty()&&producer_done)break;
                    work=move(frames.back());
                    frames.pop_back();
                }
                auto& frm=*work;
                ostringstream title;
                title<<" frame id: "<<frm.id<<" fps: "<<(high_resolution_clock::now()-frm.s).count();
                setWindowTitle("video",title.str());
                imshow("video",frm.piece);
                waitKey(1);
                frames_ready.notify_one();
                
            }
            frames_ready.notify_all();

    

}

void dater(VideoCapture &v, vector<frame>&frames )
{



  while(true){
    Mat temp;
    if(!v.read(temp))break;
    {
          unique_lock<mutex> locker(mtx);
  frames_ready.wait(locker,[&]{return frames.size()<30|| producer_done;  });
  if(producer_done)break;

  frames.emplace_back(move(temp));
    }
  frames_ready.notify_one();

}
{
    lock_guard<mutex>lockl(mtx);
 producer_done=true;
}  
  frames_ready.notify_all();
}

int main()
{
    VideoCapture v(0);
    vector<frame> imag;
    if(!v.isOpened())
    {
        cerr<<" error accessing camera";
        return -1;
    }
    v.set(CAP_PROP_FRAME_WIDTH,1280);
v.set(CAP_PROP_FRAME_HEIGHT,720);   
    thread t1(dater,ref(v),ref(imag));
   stream(imag);
    t1.join();
    return 0;


}