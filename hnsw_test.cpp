#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"
#include <unordered_set>
#include <string>
#include "sys/time.h"
#include <map>
#include <utility>

using namespace std;
using namespace hnswlib;


vector<string> split(const string& line,string flag){
    vector<string> ret;
    int start_pos = 0;
    int pos = line.find(flag);
    while(pos != string::npos){
       ret.push_back(line.substr(start_pos,pos - start_pos));
       start_pos = pos + flag.length();
       pos = line.find(flag,start_pos);
    }
    if(start_pos < line.length()){
       ret.push_back(line.substr(start_pos));
    }
    return ret;
}

int64_t GetCurrentTimeUs(){
   struct timeval tv;
   gettimeofday(&tv, 0);
   return tv.tv_sec * 1000000 + tv.tv_usec;
}

int main(){
   string vec_dim = "0.064111,-0.005155,-0.069781,-0.030958,-0.043293,0.020966,0.078878,-0.019688,0.029032,-0.073491,0.077097,-0.067048,-0.035537,-0.009496,0.052262,-0.018685,-0.024260,-0.042368,-0.025361,-0.037868,0.014875,-0.025919,0.012701,-0.086785,-0.032169,-0.027020,-0.006169,0.016283,0.001182,0.016829,0.001897,-0.004064,-0.008371,-0.028726,-0.075126,-0.053666,-0.059262,-0.049357,0.067586,-0.026346,0.050719,0.018166,0.053709,-0.028247,-0.035735,-0.020263,-0.002516,-0.003223,-0.005605,0.035986,-0.021767,-0.022036,0.020259,0.000280,0.076480,0.006846,-0.034315,0.017169,-0.118005,-0.028368,0.016233,0.056593,0.007321,-0.051076,0.027553,-0.013438,-0.031207,0.003849,0.238737,0.132774,-0.017419,0.100447,-0.116498,0.049044,0.236131,-0.227889,-0.197254,-0.036477,-0.022523,-0.133183,-0.019993,-0.193049,-0.017886,0.047218,-0.005515,-0.032679,-0.095066,-0.014187,-0.137223,-0.140570,-0.164318,-0.032249,0.042840,0.108286,0.158249,-0.015233,0.034543,0.116115,-0.019288,-0.121068,0.039469,-0.144426,0.181378,-0.283063,-0.060512,-0.077473,-0.011582,0.137924,0.097069,0.049676,-0.023586,0.097870,-0.091351,-0.142450,-0.082107,-0.175865,0.162386,0.014430,0.012647,0.297078,-0.014961,-0.082730,-0.124135,-0.133403,-0.044070,-0.033082,-0.136416,0.020136";
   int subset_size_milllions = 200;
	int efConstruction = 64;
	int M = 64;
   size_t vecdim = 128;
   int search_top = 100;
	L2Space l2(vecdim);
   size_t vecsize = subset_size_milllions * 1000000;
   string file_name = "embedding.txt";
   size_t qsize = 300*10000;
   HierarchicalNSW<float>* graph = new HierarchicalNSW<float>(&l2,qsize,M,efConstruction);
   ifstream file("../"+file_name,ios::in);
   if(!file.is_open()){
      std::cout<<"not open!!"<<endl;
      return 0;
   }
   string line;
   vector<std::pair<unsigned char*,int>> data;
   int label = 0;
   int64_t start_time = GetCurrentTimeUs();
   while(getline(file,line)){
      unsigned char* buf = new unsigned char[sizeof(float)*vecdim];
      vector<string> v = split(line,",");
      for(int i=0;i<v.size();++i){
         float f = atof(v[i].c_str());
         memcpy(buf+(sizeof(float)*i),&f,sizeof(float));
      }
      graph->addPoint(buf,label);
      data.emplace_back(buf,label);
      label++;
   }
   std::cout<<file_name<<endl;
   std::cout<<"ef:"<<efConstruction<<",M:"<<M<<endl;
   std::cout<<"build_graph_time:"<<(GetCurrentTimeUs() - start_time)/1000.0<<endl;
   vector<string> input = split(vec_dim,",");
   unsigned char input_vec[sizeof(float)*vecdim];
   for(int i=0;i<input.size();++i){
      float f = atof(input[i].c_str());
      memcpy(input_vec+(sizeof(float)*i),&f,sizeof(float));
   }
   start_time = GetCurrentTimeUs();
   auto search_ret = graph->searchKnnCloserFirst(input_vec,search_top);
   std::cout<<"search_graph_time:"<<(GetCurrentTimeUs() - start_time)/1000.0<<endl;
   // for(auto & i : search_ret){
   //    std::cout<<"dis:"<<i.first<<"|label:"<<i.second<<endl;
   // }

   struct compare_{
      bool operator()(std::pair<float,int> a, pair<float,int> b){
         return a.first > b.first; //more near more small
      }
   };
   start_time = GetCurrentTimeUs();
   std::priority_queue<pair<float,int>,vector<pair<float,int>>,compare_> force_search;
   for(int i=0;i<data.size();++i){
      DISTFUNC<float> fst_ = l2.get_dist_func();
      force_search.push(std::make_pair(fst_(input_vec,data[i].first,&vecdim),i));
   }
   std::cout<<"force_search_time:"<<(GetCurrentTimeUs() - start_time)/1000.0<<endl;
   std::cout<<"force_search_result"<<endl;
   std::map<int,float> force_;
   for(int i=0;i<search_top;++i){
      pair<float,int> t = force_search.top();
      force_[t.second] = t.first;
      //std::cout<<"dis:"<<t.first<<"|label:"<<t.second<<endl;
      force_search.pop();
   }
   int sum = 0;
   for(int i=0;i<search_ret.size();++i){
      int lab = search_ret[i].second;
      if(force_.count(lab)){
         sum++;
      }
   }
   std::cout<<"precise:"<<sum/(float)search_top<<"%"<<endl;
   return 0; 
}