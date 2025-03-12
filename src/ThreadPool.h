#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <stdexcept> 

/**
 * Thread pool for parallel computation
 */
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    /**
     * Creates a thread pool with the specified number of threads
     */
    ThreadPool(size_t numThreads);

    /**
     * Enqueues a task to be executed by the thread pool
     */
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    /**
     * Gets the number of pending tasks
     */
    size_t pendingTasks();

    /**
     * Destructor - waits for all threads to finish
     */
    ~ThreadPool();
};

#endif // THREADPOOL_H