#include "libvol/calib/svi_slice.hpp"
#include "libvol/models/vol_surface.hpp"
#include "libvol/models/black_scholes.hpp"

#include <iostream>
#include <vector>
#include <random>

int main(){
    double S = 50.0, r = 0.05, q = 0.01;
    double a = 0.48, b = 0.52;
    bool is_call = true;
    std::vector<double> Ts = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> Ks = {47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0};
    size_t nKs = Ks.size();
    size_t nTs = Ts.size();
    std::vector<std::vector<vol::OptionSpec>> opts(nTs, std::vector<vol::OptionSpec>(nKs));//2d container for slices
    std::vector<std::vector<double>> mids(nTs, std::vector<double>(nKs)); //2d container for mids
    std::vector<vol::surface::MaturitySlice> slices(nTs);//one slice per T
    std::random_device rd; //random seed for vol
    std::mt19937 gen(rd()); //random num for vol
    std::uniform_real_distribution<double> dist(a, b); //random volatility dist between lower and higher bound
    double randomvol;

    for (double& t : Ts) {
    t /= 365.0;
    }
    for (size_t i = 0; i < nTs; ++i){
        for (size_t j = 0; j < nKs; ++j){
            randomvol = dist(gen);
            opts[i][j] = {S, Ks[j], r, q, Ts[i], is_call};//2d array of synthetic opts for surface calib
            //synthetic mids with black-scholes and random vol between 0.48 and 0.52
            mids[i][j] = vol::bs::price(S, Ks[j], r, q, Ts[i], randomvol, is_call);
        }
        slices[i].options = opts[i];
        slices[i].mids = mids[i];
    }
    vol::svi::SliceConfig cfg;
    vol::surface::Surface surf;

    surf = vol::surface::calibrate_surface(slices, cfg);
    double iv;
    double T_days;

    std::cout << "Calibrated implied vol surface (per maturity)\n\n";
    for (size_t i = 0; i < nTs; ++i){
        T_days = Ts[i] * 365.0;
        std::cout << "T = " << T_days << " days\n";
        std::cout << " K     impliedvol\n";
        for (size_t j = 0; j < nKs; ++j){
            iv = surf.implied_vol(S, Ks[j], r, q, Ts[i]);
            std::cout << Ks[j] << "         " << iv << "\n";
        }
        std::cout << "\n";
    }
    return 0;
}