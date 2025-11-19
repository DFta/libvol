#include "libvol/math/quadrature.hpp"
#include "libvol/mc/gbm.hpp"
#include "libvol/models/black_scholes.hpp"
#include "libvol/models/binom.hpp"
#include "libvol/models/heston.hpp"
#include "libvol/models/svi.hpp"
#include <iostream>
#include <cctype>
#include <limits>

int main(){
    double S, K, r, q, T, vol;
    double iv_price;
    bool is_call, is_american;
    char test, euroamer;
    int type, binom_steps, mc_paths;
    std::string optiontype;
    vol::bs::PriceGreeks bs_results;
    vol::binom::PriceGreeks binom_results;
    vol::bs::IVResult iv_results;
    vol::mc::MCResult mc_results;
    vol::heston::Params;

    std::cout << "Input  Pricing engine\n";
    std::cout << "  1    Black-Scholes\n";
    std::cout << "  2    Binom\n";
    std::cout << "  3    Monte Carlo\n";
    std::cout << "  4    Heston\n";
    std::cout << "  5    All\n";
    std::cout << "----------------------\n";
    std::cout << "  6    Simple IV\n";
    while(true){
        std::cout << "> ";
        std::cin >> type;
        if (type == 1 || type == 2 || type == 3 || type == 4 || type == 5 || type == 6) break;
        std::cout << "Not a type. \n";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    if (type == 4) {
        std::cout << "Not implemented yet sowwy:(\n";
        std::cout << "Enter to quit ";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cin.get();
        return 0;
    }
    if (type == 2 || type == 3 || type == 5){
        if (type == 2 || type == 5){    
            while (true) {
            std::cout << "Euro or american? (e/a): ";
            if (std::cin >> euroamer && (euroamer == 'e' || euroamer == 'a')) break;
            std::cout << "not an option type.\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
            is_american = (euroamer == 'a');
        }
        while(true){
            std::cout << "Number of steps ";
            (type == 5) ? std::cout << "for Binom: " : std::cout << ": ";
            if (std::cin >> binom_steps && binom_steps >= 0) break;
            std::cout << "Please input a positive integer.\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
    if (type == 3 || type == 5){
        while(true){
            std::cout << "Number of paths ";
            (type == 5) ? std::cout << "for MC: " : std::cout << ": ";
            if (std::cin >> mc_paths && mc_paths >= 0) break;
            std::cout << "Please input a positive integer.\n";
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
    while (true) {
        std::cout << "Input your own specs? y/n ";
        if (std::cin >> test && (test == 'y' || test == 'n')) {
            break;
        }
        std::cout << "Please enter 'y' or 'n'.\n";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    if (test == 'y'){
        do{
        std::cout << "call or put?: ";
        std::cin >> optiontype;
        if (optiontype != "call" && optiontype != "put" && optiontype != "Call" && optiontype != "Put") std::cout << "not an option type. \n";
        } while (optiontype != "call" && optiontype != "put" && optiontype != "Call" && optiontype != "Put");
        is_call = (optiontype == "call" || optiontype == "Call") ? true : false;
        std::cout << "Enter current underlying price: ";
        std::cin >> S;
        std::cout << "Enter option strike price: ";
        std::cin >> K;
        std:: cout << "Enter current risk-free rate: ";
        std::cin >> r;
        std::cout << "Enter dividend yield (0 if none): ";
        std:: cin >> q;
        std::cout << "Enter time to expiration, days: ";
        std::cin >> T;
        T /= 365.0;
        if (type == 6){
            std::cout << "Enter current option price: ";
            std::cin >> iv_price;
        } else {
        std::cout << "Enter volatility: ";
        std::cin >> vol;
        }   
    } else {S = 100.0, K = 100.0, r = 0.05, q = 0.01, T = 1, vol = 0.25, is_call = true;}

    if (type == 1 || (type == 5 && !is_american)){
        if (type == 5){
            std::cout << "Black-Scholes: \n";
        }
        bs_results = vol::bs::price_greeks(S,K,r,q,T,vol,is_call);
        std::cout << "Price: " << bs_results.price << "\n";
        std::cout << "Delta: " << bs_results.delta << "\n";
        std::cout << "Gamma: " << bs_results.gamma << "\n";
        std::cout << "Rho  : " << bs_results.rho/100.0 << "\n";
        std::cout << "Theta: " << bs_results.theta/365.0 << "\n";
        std::cout << "Vega : " << bs_results.vega/100.0 << "\n";
    } 
    if (type == 2 || type == 5){
        if (type == 5){
            std::cout << "-----------------\n";
            std::cout << "Binomial: \n";
        }
        binom_results = vol::binom::price_greeks(S,K,r,q,T,vol,binom_steps,is_call,is_american);
        std::cout << "Price: " << binom_results.price << "\n";
        std::cout << "Delta: " << binom_results.delta << "\n";
        std::cout << "Gamma: " << binom_results.gamma << "\n";
        std::cout << "Rho  : " << binom_results.rho/100.0 << "\n";
        std::cout << "Theta: " << binom_results.theta/365.0 << "\n";
        std::cout << "Vega : " << binom_results.vega/100.0 << "\n";
    } 
    if (type == 3 || type == 5){
        if (type == 5){
            std::cout << "-----------------\n";
            std::cout << "Monte-Carlo: \n";
        }
        mc_results = vol::mc::european_vanilla_gbm(S,K,r,q,T,vol,is_call,mc_paths);
        std::cout << "Price    : " << mc_results.price << "\n";
        std::cout << "Std Error: " << mc_results.std_err << "\n";
        iv_results = vol::bs::implied_vol(S,K,r,q,T,iv_price,is_call);
    } 
    if (type == 4){
        std::cout << "Not implemented yet sowwy:(\n";
    } else if (type == 6){
        iv_results = vol::bs::implied_vol(S,K,r,q,T,iv_price,is_call);
        if (iv_results.converged){
        std::cout << "Implied volatility: " << iv_results.iv << "\n";
        } else {std::cout << "Solver did not converge.\n";}
    }
    std::cout << "Enter to quit ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
    return 0;
}