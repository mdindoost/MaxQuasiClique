#include "ThreadPool.h"

using namespace std;

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                function<void()> task;
                
                {
                    unique_lock<mutex> lock(queue_mutex);
                    condition.wait(lock, [this] { return stop || !tasks.empty(); });
                    
                    if (stop && tasks.empty()) {
                        return;
                    }
                    
                    task = move(tasks.front());
                    tasks.pop();
                }
                
                task();
            }
        });
    }
}

size_t ThreadPool::pendingTasks() {
    unique_lock<mutex> lock(queue_mutex);
    return tasks.size();
}

ThreadPool::~ThreadPool() {
    {
        unique_lock<mutex> lock(queue_mutex);
        stop = true;
    }
    
    condition.notify_all();
    
    for (thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}