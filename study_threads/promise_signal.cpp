#include <chrono>
#include <future>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>

// Scenario: Background thread performs a blocking I/O task and communicates both
// success/failure via std::promise.

struct WeatherReport
{
    std::string city;
    double temperature_c{};
    bool success{};
    std::string message;
};

WeatherReport fetch_weather(const std::string& city, std::chrono::milliseconds simulated_latency)
{
    std::this_thread::sleep_for(simulated_latency);
    WeatherReport report;
    report.city = city;

    static std::mt19937 rng{42};
    std::uniform_real_distribution<double> temp_dist{-10.0, 40.0};
    report.temperature_c = temp_dist(rng);
    report.success = true;
    report.message = "OK";
    return report;
}

void weather_service(std::string city, std::promise<WeatherReport> promise)
{
    try {
        // TODO: Inject failure paths (throw, set_exception) to understand propagation.
        auto report = fetch_weather(std::move(city), std::chrono::milliseconds(350));
        promise.set_value(std::move(report));
    } catch (...) {
        promise.set_exception(std::current_exception());
    }
}

int main()
{
    std::promise<WeatherReport> report_promise;
    std::future<WeatherReport> report_future = report_promise.get_future();

    std::thread worker(weather_service, "Lisbon", std::move(report_promise));

    // TODO: Replace get() with wait_for loop to implement a timeout fallback.
    try {
        auto report = report_future.get();
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "[promise_signal] " << report.city << " = " << report.temperature_c << "°C\n";
    } catch (const std::exception& ex) {
        std::cout << "[promise_signal] future delivered exception: " << ex.what() << '\n';
    }

    worker.join();

    // TODO: Add multiple consumers via std::shared_future and observe read-only behavior.
    return 0;
}

