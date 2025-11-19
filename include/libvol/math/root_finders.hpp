#pragma once
#include <functional>
#include <stdexcept>
#include <cmath>
#include <limits>


namespace vol::root {
struct Result { double x; int iters; bool converged; };


inline Result newton(std::function<void(double,double&,double&)> f_df,

    double x0, double tol=1e-10, int maxit=50) {
        double x=x0; 
        double f=0, 
        df=0;
        for(int i=0;i<maxit;++i){
            f_df(x,f,df);
            if(std::abs(df) < 1e-14) break;
            double step = f/df;
            x -= step;
            if(std::abs(step) < tol*(1.0+std::abs(x))) {
                return {x,i+1,true};
            }
        }
        return {x,maxit,false};
    }


    inline Result brent(std::function<double(double)> f, double a, double b, double tol=1e-10, int maxit=100){
        double fa=f(a), fb=f(b);
        if(fa*fb>0) {
            throw std::invalid_argument("brent: root not bracketed");
        }
        double c=a, fc=fa; bool mflag=true; double s=b; double d=0;
        for(int iter=1; iter<=maxit; ++iter){
            if(std::abs(fc) < std::abs(fb)){ 
                std::swap(a,b); 
                std::swap(fa,fb); 
                std::swap(c,b); 
                std::swap(fc,fb);
            }
            double tol1 = 2*std::numeric_limits<double>::epsilon()*std::abs(b) + tol/2;
            double m = 0.5*(c-b);
            if(std::abs(m) <= tol1 || fb==0.0) return {b,iter,true};
            if(std::abs(fa - fc) > 0 && std::abs(fb - fa) > 0){
                s = a*fb*fc/((fa-fb)*(fa-fc)) + b*fa*fc/((fb-fa)*(fb-fc)) + c*fa*fb/((fc-fa)*(fc-fb));
            } else {
                s = b - fb*(b-a)/(fb-fa);
            }
            double cond1 = (s < (3*a+b)/4 || s > c);
            double cond2 = (mflag && std::abs(s-b) >= std::abs(b-c)/2);
            double cond3 = (!mflag && std::abs(s-b) >= std::abs(c-d)/2);
            double cond4 = (mflag && std::abs(b-c) < tol1);
            double cond5 = (!mflag && std::abs(c-d) < tol1);
            if(cond1 || cond2 || cond3 || cond4 || cond5){ 
                s=(a+b)/2; mflag=true; 
            }
            else mflag=false;
            d=c; 
            c=b; 
            fc=fb;
            double fs=f(s); a=((fb*fs)<0) ? b : a; 
            b=s; 
            fb=fs;
            if(a>b){ 
                std::swap(a,b); 
                std::swap(fa,fb);
            } else { 
                fa=f(a);
            }
        }
        return {b,maxit,false};
    }
} // namespace vol::root