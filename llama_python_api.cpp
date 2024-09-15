#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "common.h"
#include "llama.h"

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace py = pybind11;

class LlamaModel {
public:
    LlamaModel(const std::string& model_path, int n_ctx = 2048, int n_threads = -1, bool use_mlock = false, bool use_mmap = true) {
        try {
            params.model = model_path;
            params.n_ctx = n_ctx;
            params.use_mlock = use_mlock;
            params.use_mmap = use_mmap;

            llama_backend_init();

            model_params = llama_model_params_from_gpt_params(params);
            model = llama_load_model_from_file(params.model.c_str(), model_params);

            if (model == NULL) {
                throw std::runtime_error("Unable to load model from " + model_path);
            }

            ctx_params = llama_context_params_from_gpt_params(params);
            ctx_params.n_threads = n_threads == -1 ? std::thread::hardware_concurrency() : n_threads;
            ctx_params.n_threads_batch = ctx_params.n_threads;

            ctx = llama_new_context_with_model(model, ctx_params);

            if (ctx == NULL) {
                llama_free_model(model);
                throw std::runtime_error("Failed to create the llama_context");
            }

            log("Model loaded successfully. n_ctx: " + std::to_string(n_ctx) + ", n_threads: " + std::to_string(ctx_params.n_threads));

            // Start worker threads
            for (int i = 0; i < ctx_params.n_threads; ++i) {
                workers.emplace_back([this] { this->worker(); });
            }
        } catch (const std::exception& e) {
            log("Error in LlamaModel constructor: " + std::string(e.what()));
            throw;
        }
    }

    ~LlamaModel() {
        try {
            log("Shutting down LlamaModel");
            // Stop worker threads
            running = false;
            cv.notify_all();
            for (auto& worker : workers) {
                worker.join();
            }

            llama_free(ctx);
            llama_free_model(model);
            llama_backend_free();
            log("LlamaModel shutdown complete");
        } catch (const std::exception& e) {
            log("Error in LlamaModel destructor: " + std::string(e.what()));
        }
    }

    std::vector<std::string> batch_process(const std::vector<std::string>& prompts, int n_predict = 128) {
        try {
            log("Starting batch processing of " + std::to_string(prompts.size()) + " prompts");
            auto start_time = std::chrono::high_resolution_clock::now();

            std::vector<std::string> results(prompts.size());
            std::atomic<size_t> next_prompt(0);
            std::atomic<size_t> completed_prompts(0);

            auto process_func = [&](size_t thread_id) {
                while (true) {
                    size_t i = next_prompt.fetch_add(1);
                    if (i >= prompts.size()) break;
                    results[i] = generate(prompts[i], n_predict);
                    completed_prompts.fetch_add(1);
                    if (completed_prompts % 10 == 0) {  // Log progress every 10 completions
                        log("Processed " + std::to_string(completed_prompts) + " out of " + std::to_string(prompts.size()) + " prompts");
                    }
                }
            };

            std::vector<std::thread> threads;
            for (int i = 0; i < ctx_params.n_threads; ++i) {
                threads.emplace_back(process_func, i);
            }

            for (auto& thread : threads) {
                thread.join();
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            log("Batch processing completed in " + std::to_string(duration.count()) + " ms");

            return results;
        } catch (const std::exception& e) {
            log("Error in batch_process: " + std::string(e.what()));
            throw;
        }
    }

    std::string generate(const std::string& prompt, int n_predict = 128) {
        try {
            std::vector<llama_token> tokens_list = ::llama_tokenize(ctx, prompt, true);

            llama_batch batch = llama_batch_init(std::min(512, (int)tokens_list.size()), 0, 1);

            for (size_t i = 0; i < tokens_list.size(); i++) {
                llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
            }

            batch.logits[batch.n_tokens - 1] = true;

            if (llama_decode(ctx, batch) != 0) {
                throw std::runtime_error("llama_decode() failed");
            }

            std::string output = prompt;
            int n_cur = batch.n_tokens;

            while (n_cur <= n_predict) {
                auto n_vocab = llama_n_vocab(model);
                auto* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);

                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
                }

                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

                if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict) {
                    break;
                }

                output += llama_token_to_piece(ctx, new_token_id);

                llama_batch_clear(batch);
                llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

                n_cur += 1;

                if (llama_decode(ctx, batch)) {
                    throw std::runtime_error("Failed to eval");
                }
            }

            llama_batch_free(batch);

            return output;
        } catch (const std::exception& e) {
            log("Error in generate: " + std::string(e.what()));
            throw;
        }
    }

    

private:
    gpt_params params;
    llama_model_params model_params;
    llama_context_params ctx_params;
    llama_model* model;
    llama_context* ctx;

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> running{true};

    void worker() {
        while (running) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this] { return !tasks.empty() || !running; });
                if (!running && tasks.empty()) return;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
        }
    }

    void add_task(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.push(std::move(task));
        }
        cv.notify_one();
    }

    void log(const std::string& message) {
        std::stringstream ss;
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << " [LlamaModel] " << message << std::endl;
        std::cout << ss.str();
    }
};

PYBIND11_MODULE(llama_cpp, m) {
    py::class_<LlamaModel>(m, "LlamaModel")
        .def(py::init<const std::string&, int, int, bool, bool>(),
             py::arg("model_path"),
             py::arg("n_ctx") = 2048,
             py::arg("n_threads") = -1,
             py::arg("use_mlock") = false,
             py::arg("use_mmap") = true)
        .def("generate", &LlamaModel::generate,
             py::arg("prompt"),
             py::arg("n_predict") = 128)
        .def("batch_process", &LlamaModel::batch_process,
             py::arg("prompts"),
             py::arg("n_predict") = 128);
}
