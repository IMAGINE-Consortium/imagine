#include <cassert>
#include <thread>
#include <chrono>
#include <ctime>
#include <memory>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

/**
 * this is a C++ module for random value/seed generation in IMAGINE
 * in Python Numpy module, the random status is a global attribute
 * a stand alone C++ module for random generation in IMAGINE is neccssary
 * for avoid unexpected interference between IMAGINE and its ported sampler, ie, Dynesty
 */
class RandModule{
public:
    RandModuel() = default;
    RandModule(const RandModule &) = delete;
    RandModule(RandModule &&) = delete;
    RandModule& operator= (RandModule &&) = delete;
    RandModule& operator= (const RandModule &) = delete;
    virtual ~RandModule () = default;
    /**
     * generate a random seed
     * if given non-zero integer, take it as the seed
     * if given zero, generate a time-thread dependent seed
     */
    virtual std::size_t seed (const std::size_t&) const;
    /**
     * generate a vector of random non-negative integers
     * given the seed for generating, if zero, do time-thread dependent seed
     * given the size of the vector
     * given lower value limit
     * given upper value limit
     */
    virtual std::unique_ptr<std::array<double>> unif_double_list (const std::size_t&,
                                                                  const std::size_t&,
                                                                  const double&,
                                                                  const double&) const;
}

// offer time-thread dependent random seed
std::size_t RandModule::generate_seed (const std::size_t& seed) const{
    assert(seed>=0);
    if(seed==0){
        auto p = std::chrono::system_clock::now();
        // valid until 19 January, 2038 03:14:08 UTC
        time_t today_time = std::chrono::system_clock::to_time_t(p);
        // casting thread id into unsinged long
        std::stringstream ss;
        ss << std::this_thread::get_id();
        auto th_id = std::stoul(ss.str());
        // precision in (thread,second)
        return (th_id + today_time);
    }
    return seed;
}

// offer an array of random floating numbers
std::unique_ptr<std::array<double>> RandModule::unif_double_list (const std::size_t& seed,
                                                                  const std::size_t& size,
                                                                  const double& low,
                                                                  const double& high) const{
    auto cache = std::make_unique<std::array<double>>(size);
    gsl_rng *r {gsl_rng_alloc(gsl_rng_taus)};
    gsl_rng_set (r, this->generate_seed(seed));
    for (std::size_t i=0;i<size;++i){
        cache[i] = gsl_rng_uniform(r)*(high-low) + low;
    }
    return cache;
}
