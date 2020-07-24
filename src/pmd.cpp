#include "pmd.h"

#include <math.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <mkl.h>
#pragma GCC diagnostic pop
#include <omp.h>
#include <zmq.h>

#include <algorithm> /* sort, max */
#include <iostream>  /* cerr */

#include "decimation.h"
#include "line_search.h"
#include "proxtv.h"
#include "welch.h"

/*----------------------------------------------------------------------------*
 *--------------------------- Generic PMD Parameter PKG ----------------------*
 *----------------------------------------------------------------------------*/

PMD_params::PMD_params(
    const MKL_INT _bheight, const MKL_INT _bwidth, MKL_INT _d_sub,
    const MKL_INT _t, MKL_INT _t_sub, const double _spatial_thresh,
    const double _temporal_thresh, const size_t _max_components,
    const size_t _consec_failures, const size_t _max_iters_main,
    const size_t _max_iters_init, const double _tol, void *_FFT,
    bool _enable_temporal_denoiser, bool _enable_spatial_denoiser)
    :

      bheight(_bheight), bwidth(_bwidth), d_sub(_d_sub), t(_t), t_sub(_t_sub),
      spatial_thresh(_spatial_thresh), temporal_thresh(_temporal_thresh),
      max_components(_max_components), consec_failures(_consec_failures),
      max_iters_main(_max_iters_main), max_iters_init(_max_iters_init),
      tol(_tol), FFT(_FFT), enable_temporal_denoiser(_enable_temporal_denoiser),
      enable_spatial_denoiser(_enable_spatial_denoiser) {}

MKL_INT PMD_params::get_bheight() { return this->bheight; }

MKL_INT PMD_params::get_bwidth() { return this->bwidth; }

MKL_INT PMD_params::get_d_sub() { return this->d_sub; }

void PMD_params::set_d_sub(MKL_INT _d_sub) { this->d_sub = _d_sub; }

MKL_INT PMD_params::get_t() { return this->t; }

MKL_INT PMD_params::get_t_sub() { return this->t_sub; }

void PMD_params::set_t_sub(MKL_INT _t_sub) { this->t_sub = _t_sub; }

double PMD_params::get_spatial_thresh() { return this->spatial_thresh; }

double PMD_params::get_temporal_thresh() { return this->temporal_thresh; }

size_t PMD_params::get_max_components() { return this->max_components; }

size_t PMD_params::get_consec_failures() { return this->consec_failures; }

size_t PMD_params::get_max_iters_main() { return this->max_iters_main; }

size_t PMD_params::get_max_iters_init() { return this->max_iters_init; }

double PMD_params::get_tol() { return this->tol; }

void *PMD_params::get_FFT() { return this->FFT; }

void PMD_params::set_FFT(void *_FFT) { this->FFT = _FFT; }

bool PMD_params::get_enable_temporal_denoiser() {
    return this->enable_temporal_denoiser;
}

void PMD_params::set_enable_temporal_denoiser(bool _enable_temporal_denoiser) {
    this->enable_temporal_denoiser = _enable_temporal_denoiser;
}

bool PMD_params::get_enable_spatial_denoiser() {
    return this->enable_spatial_denoiser;
}

void PMD_params::set_enable_spatial_denoiser(bool _enable_spatial_denoiser) {
    this->enable_spatial_denoiser = _enable_spatial_denoiser;
}

/*----------------------------------------------------------------------------*
 *--------------------------- Spatial Helper Funcs ---------------------------*
 *----------------------------------------------------------------------------*/

/* Computes the ratio of the L1 to TV norm for a spatial components in order to
 * test against the null hypothesis that it is gaussian noise.
 */
double estimate_noise_tv_op(const MKL_INT d1, const MKL_INT d2,
                            const double *u_k) {
    int j, j1, j2;
    double std;
    int n = 0;

    int num_edges = d1 * (d2 - 1) + d2 * (d1 - 1);
    std::vector<double> edge_diffs(num_edges);

    /* All Elements Except Union (Bottom Row, Rightmost Column) */
    for (j2 = 0; j2 < d2 - 1; j2++) {
        for (j1 = 0; j1 < d1 - 1; j1++) {
            j = d1 * j2 + j1;
            edge_diffs[n] = fabs(u_k[j] - u_k[j + 1]);
            n++;
            edge_diffs[n] = fabs(u_k[j] - u_k[j + d1]);
            n++;
        }
    }

    /* Rightmost Column (Skip Bottom Element) */
    j = d1 * (d2 - 1);
    for (j1 = 0; j1 < d1 - 1; j1++) {
        edge_diffs[n] = fabs(u_k[j] - u_k[j + 1]);
        n++;
        j += 1;
    }

    /* Bottom Row (Skip Rightmost Element) */
    j = d1 - 1;
    for (j2 = 0; j2 < d2 - 1; j2++) {
        edge_diffs[n] = fabs(u_k[j] - u_k[j + d1]);
        n++;
        j += d1;
    }

    /* Sort Differences */
    std::sort(&edge_diffs[0], &edge_diffs[num_edges]);

    /* Transform & Return */
    std = edge_diffs[num_edges / 2] / .954;
    return std * std;
}

/* Estimates Image Noise Variance By Taking Median Of Local Noise Var Estimates
 * */
double estimate_noise_mean_filter(const int rows, const int cols,
                                  double *image) {
    int n = 5;
    double var_hat;
    std::vector<double> mses(rows * cols, 0.0);

    /* Visit Each Pixel */
    int r;
    for (r = 0; r < rows; r++) {
        int c;
        for (c = 0; c < cols; c++) {
            /* Extract Window Around Pixels */
            double err, mean = 0;
            int rr, cc, p = 0;
            for (rr = (r - (n / 2)); rr < (r - (n / 2) + n); rr++) {
                for (cc = (c - (n / 2)); cc < (c - (n / 2) + n); cc++) {
                    if ((rr >= 0) && (rr < rows) && (cc >= 0) && (cc < cols)) {
                        mean += image[rr + rows * cc];
                        p++;
                    }
                }
            }

            /* Compute MSE At Pixel [r, c] */
            mean /= p;
            err = image[r + rows * c] - mean;
            mses[r + rows * c] = err * err;
        }
    }

    /* Sort Local MSES & Return Median */
    std::sort(mses.begin(), mses.end());
    var_hat = mses[(rows * cols) / 4];
    return var_hat;
}

/* Apply Median Filter To Estimate Noise Level
 */
double estimate_noise_median_filter(const int rows, const int cols,
                                    double *image) {
    int n = 3;
    double var_hat;
    std::vector<double> mses(rows * cols, 0.0);
    std::vector<double> pixel_values(n * n, 0.0);

    /* Visit Each Pixel */
    int r;
    for (r = 0; r < rows; r++) {
        int c;
        for (c = 0; c < cols; c++) {
            /* Extract Window Around Pixels */
            double err, median = 0;
            int rr, cc, p = 0;
            for (rr = (r - (n / 2)); rr < (r - (n / 2) + n); rr++) {
                for (cc = (c - (n / 2)); cc < (c - (n / 2) + n); cc++) {
                    if ((rr >= 0) && (rr < rows) && (cc >= 0) && (cc < cols)) {
                        pixel_values[p] = image[rr + rows * cc];
                        p++;
                    }
                }
            }

            /* Sort Pixel Values In Window To Find Median */
            std::sort(&pixel_values[0], &pixel_values[p]);
            median = pixel_values[p / 2];
            err = image[r + rows * c] - median;
            mses[r + rows * c] = err * err;
        }
    }

    /* Sort Local MSES & Return Median */
    std::sort(mses.begin(), mses.end());
    var_hat = mses[(rows * cols) / 2];
    return var_hat;
}

/* Solve Constrained TV With CPS Line Search */
int cps_tv(const int d1, const int d2, double *y, const double delta, double *x,
           double *lambda_tv, double *info, double tol) {
    int d = d1 * d2;
    int iters = 0;
    double target = sqrt(d * delta); // target norm of error
    double l2_err, l2_err_prev = 0;
    std::vector<double> resid(d, 0.0);

    while (iters < 100) {
        /* Evaluate Solution At Lambda */
        DR2_TV(d1, d2, y, *lambda_tv, *lambda_tv, 1, 1, x, 1, 1, info);

        /* Compute norm of residual */
        vdSub(d, y, x, &resid[0]);
        l2_err = cblas_dnrm2(d, &resid[0], 1);

        /* Check For Convergence */
        if (fabs(target * target - l2_err * l2_err) / (target * target) < tol) {
            return 1; // successfully converged within tolerance
        } else if (fabs(l2_err - l2_err_prev) < 1e-3) {
            return 0; // line search stalled
        }
        l2_err_prev = l2_err;

        /* Increment Lambda */
        // *lambda_tv = exp(log(*lambda_tv) + log(target) - log(l2_err));
        *lambda_tv = (*lambda_tv) * target / l2_err;

        /* Update Iteration Count */
        iters++;
    }
    return 0; // Line Search Didn't Converge
}

/* Denoises and normalizes the  d1xd2 spatial component u_k using the
 * proxTV douglass rachford splitting cpp implementation of TV denoising.
 */
void constrained_denoise_spatial(const MKL_INT d1, const MKL_INT d2,
                                 double *u_k, double *lambda_tv) {
    int status;
    double delta;

    std::vector<double> target(d1 * d2, 0.0);
    std::vector<double> info(3, 0.0);

    copy(d1 * d2, u_k, &target[0]);

    /* Compute Noise Level */
    delta = estimate_noise_mean_filter(d1, d2, u_k);
    /*delta = estimate_noise_median_filter(d1,d2,u_k);*/
    if (delta > 0) {
        /* u_k <- argmin_u ||u_k - u|| + 2*lambda_tv ||u||TV */
        status = cps_tv(d1, d2, &target[0], delta, u_k, lambda_tv, &info[0]);
        if (status < 1) {
            *lambda_tv = .0025;
            DR2_TV(d1, d2, &target[0], *lambda_tv, *lambda_tv, 1, 1, u_k, 1, 1,
                   &info[0]);
        }

        /* u_k /= ||u_k|| */
        normalize(d1 * d2, u_k);
    }
}

/* Denoises and normalizes the  d1xd2 spatial component u_k using the
 * proxTV douglass rachford splitting cpp implementation of TV denoising.
 */
void denoise_spatial(const MKL_INT d1, const MKL_INT d2, double *u_k,
                     double *lambda_tv) {
    std::vector<double> target(d1 * d2, 0.0);
    std::vector<double> info(3, 0.0);

    copy(d1 * d2, u_k, &target[0]);

    /* u_k <- argmin_u ||u_k - u|| + 2*lambda_tv ||u||TV */
    DR2_TV(d1, d2, &target[0], *lambda_tv, *lambda_tv, 1, 1, u_k, 1, 1,
           &info[0]);

    /* u_k /= ||u_k|| */
    normalize(d1 * d2, u_k);
}

/* Update the (possibly downsampled) spatial component by regression of the
 * temporal component against the residual followed by normalization.
 * Returns the normed difference between the updated spatial component
 * and the previous iterate (used to monitor convergence).
 */
double update_spatial_init(const MKL_INT d, const MKL_INT t, const double *R_k,
                           double *u_k, const double *v_k) {
    double delta_u;
    std::vector<double> u__(d);

    /* u__ <- u_k */
    copy(d, u_k, &u__[0]);

    /* u_{k+1} <- R_{k+1} v_k / ||R_{k+1} v_k||_2 */
    regress_spatial(d, t, R_k, u_k, v_k);

    /* delta_u <- ||u_{k+1} - u_{k}||_2 */
    delta_u = distance_inplace(d, u_k, &u__[0]);

    return delta_u;
}

/* Update spatial component by regression of temporal component against
 * the residual followed by denoising via TV prox operator. Returns the
 * normed difference between the updated spatial component and the
 * previous iterate (used to monitor convergence).
 */
double update_spatial(const double *R_k, double *u_k, const double *v_k,
                      double *lambda_tv, PMD_params *pars) {
    MKL_INT d1 = pars->get_bheight();
    MKL_INT d2 = pars->get_bwidth();
    MKL_INT t = pars->get_t();
    bool enable_spatial_denoiser = pars->get_enable_spatial_denoiser();

    /* Declare & Allocate For Internal Vars */
    MKL_INT d = d1 * d2;
    double delta_u;

    std::vector<double> u__(d);

    /* u__ <- u_k */
    copy(d, u_k, &u__[0]);

    /* u_{k+1} <- R_{k+1} v_k */
    regress_spatial(d, t, R_k, u_k, v_k);

    /* u_{k+1} <- argmin_u ||u_{k+1} - u||_2^2 + 2* lambda_tv ||u||_TV */
    if (enable_spatial_denoiser) {
        // enable to skip this step
        constrained_denoise_spatial(d1, d2, u_k, lambda_tv);
    }

    /* delta_u <- ||u_{k+1} - u_{k}||_2 */
    delta_u = distance_inplace(d, u_k, &u__[0]);

    return delta_u;
}

/*----------------------------------------------------------------------------*
 *--------------------------- Temporal Helper Funcs --------------------------*
 *----------------------------------------------------------------------------*/

/* Denoises and normalizes the len t spatial component v_k using the
 * constrained PDAS implementation of 2nd order L1TF denoising.
 */
void denoise_temporal(const MKL_INT t, double *v_k, double *z_k,
                      double *lambda_tf, void *FFT) {
    int iters;
    int status;
    double scale, delta;
    std::vector<double> wi(t, 1.0); // init with constant weight for now
    std::vector<double> v(t, 0.0);

    iters = 0;

    /* v <- v_k */
    copy(t, v_k, &v[0]); // target for tf

    /* Estimate Noise Level Via Welch PSD Estimate & Compute Ideal Step Size */
    delta = psd_noise_estimate(t, v_k, FFT);
    // scale = compute_scale(t, v_k, delta);
    // tau = (log(20 + (1 / scale)) - log(3 + (1 / scale))) / 60;

    /* If Uninitialized Compute Starting Point For Search */
    if (*lambda_tf <= 0) {
        scale = compute_scale(t, v_k, delta);
        *lambda_tf = exp((log(20 + (1 / scale)) - log(3 + (1 / scale))) / 2 +
                         log(3 * scale + 1)) -
                     1;
    }

    /* v_k <- argmin_{v_k} ||v_k||_TF s.t. ||v - v_k||_2^2 <= T * delta */
    // status = line_search(t, v, &wi[0], delta, tau, v_k, z_k, lambda_tf,
    // &iters, 1, 1e-3, 0);
    status =
        cps_tf(t, &v[0], &wi[0], delta, v_k, z_k, lambda_tf, &iters, 2e-2, 0);

    if (status < 1) {
        copy(t, &v[0], v_k);
    }

    /* v_k /= ||v_k||_2 */
    normalize(t, &v_k[0]);
}

/* Update the (possibly downsampled) temporal component by regression of the
 * spatial component against the residual followed by normalization.
 * Returns the normed difference between the updated temporal component
 * and the previous iterate (used to monitor convergence).
 */
double update_temporal_init(const MKL_INT d, const MKL_INT t, const double *R_k,
                            const double *u_k, double *v_k) {
    double delta_v;
    std::vector<double> v__(t, 0.0);

    /* v__ <- v_k */
    copy(t, v_k, &v__[0]);

    /* v_{k+1} <- R_{k+1}' u_k / ||R_{k+1}' u_k||_2 */
    regress_temporal(d, t, R_k, u_k, v_k);
    normalize(t, v_k);

    /* return ||v_{k+1} - v_{k}||_2 */
    delta_v = distance_inplace(t, v_k, &v__[0]);

    return delta_v;
}

/* Update temporal component by regression of spatial component against
 * the residual followed by denoising via constrained TF. Returns the
 * normed difference between the updated temporal component and the
 * previous iterate (used to monitor convergence).
 */
double update_temporal(const MKL_INT d, const double *R_k, const double *u_k,
                       double *v_k, double *z_k, double *lambda_tf,
                       PMD_params *pars) {
    double delta_v;
    MKL_INT t = pars->get_t();
    void *FFT = pars->get_FFT();
    bool enable_temporal_denoiser = pars->get_enable_temporal_denoiser();
    std::vector<double> v__(t);

    /* v__ <- v_k */
    copy(t, v_k, &v__[0]);

    /* v_{k+1} <- R_{k+1}' u_k */
    regress_temporal(d, t, R_k, u_k, v_k);

    /* v_{k+1} <- argmin_v ||v||_TF s.t. ||v_{k+1} - v||_2^2 <= T * delta */
    if (enable_temporal_denoiser) {
        denoise_temporal(t, v_k, z_k, lambda_tf, FFT);
    }

    /* return ||v_{k+1} - v_{k}||_2 */
    delta_v = distance_inplace(t, v_k, &v__[0]);

    return delta_v;
}

/*----------------------------------------------------------------------------*
 *--------------------------- Inner Loop Funcions ----------------------------*
 *----------------------------------------------------------------------------*/

/* Computes the ratio of the L1 to TV norm for a spatial components in order to
 * test against the null hypothesis that it is gaussian noise.
 */
double spatial_test_statistic(const MKL_INT d1, const MKL_INT d2,
                              const double *u_k) {
    int j, j1, j2;
    double norm_tv = 0;
    double norm_l1 = u_k[d1 * d2 - 1]; // Bottom, Right Corner

    /* All Elements Except Union (Bottom Row, Rightmost Column) */
    for (j2 = 0; j2 < d2 - 1; j2++) {
        for (j1 = 0; j1 < d1 - 1; j1++) {
            j = d1 * j2 + j1;
            norm_tv += fabs(u_k[j] - u_k[j + 1]) + fabs(u_k[j] - u_k[j + d1]);
            norm_l1 += fabs(u_k[j]);
        }
    }

    /* Rightmost Column (Skip Bottom Element) */
    j = d1 * (d2 - 1);
    for (j1 = 0; j1 < d1 - 1; j1++) {
        norm_tv += fabs(u_k[j] - u_k[j + 1]);
        norm_l1 += fabs(u_k[j]);
        j += 1;
    }

    /* Bottom Row (Skip Rightmost Element) */
    j = d1 - 1;
    for (j2 = 0; j2 < d2 - 1; j2++) {
        norm_tv += fabs(u_k[j] - u_k[j + d1]);
        norm_l1 += fabs(u_k[j]);
        j += d1;
    }

    // return norm_tv / (d1 * (d2 - 1) + d2 * (d1- 1));
    /* Return Test Statistic */
    if (norm_tv > 0) {
        return (norm_tv * (d1 * d2)) /
               (norm_l1 * (d1 * (d2 - 1) + d2 * (d1 - 1)));
    }

    return 100; // Spatial component constant => garbage
}

/* Computes the ratio of the L1 to TV norm for a spatial components in order to
 * test against the null hypothesis that it is gaussian noise.
 */
double temporal_test_statistic(const MKL_INT t, const double *v_k) {
    /* Declare & Initialize Internal Variables */
    int i;
    double norm_tf = 0;
    double norm_l1 = fabs(v_k[0]) + fabs(v_k[t - 1]);

    /* All Elements Except Union (Bottom Row, Rightmost Column) */
    for (i = 1; i < t - 1; i++) {
        norm_tf += fabs(v_k[i] + v_k[i] - v_k[i - 1] - v_k[i + 1]);
        norm_l1 += fabs(v_k[i]);
    }

    /* Return Test Statistic */
    if (norm_l1 > 0) {
        return norm_tf / norm_l1;
    }

    return 100; // temporal component constant => garbage
}

/* Solves:
 *  u_k, v_k : min <u_k, R_k v_k> - lambda_tv ||u_k||_TV - lambda_tf ||v_k||_TF
 *
 *  Returns:
 *      1: If we reject the null hypothesis that u_k is noise
 *     -1: If we accept the null hypothesis that u_k is noise
 */
int rank_one_decomposition(const double *R_k, const double *R_init, double *u_k,
                           double *u_init, double *v_k, double *v_init,
                           PMD_params *pars) {
    MKL_INT d1 = pars->get_bheight();
    MKL_INT d2 = pars->get_bwidth();
    MKL_INT d_sub = pars->get_d_sub();
    MKL_INT t = pars->get_t();
    MKL_INT t_sub = pars->get_t_sub();
    double spatial_thresh = pars->get_spatial_thresh();
    double temporal_thresh = pars->get_temporal_thresh();
    ssize_t max_iters_main = pars->get_max_iters_main();
    ssize_t max_iters_init = pars->get_max_iters_init();
    double tol = pars->get_tol();
    // void* FFT = pars->get_FFT();

    /* Declare, Allocate, & Initialize Internal Vars */
    MKL_INT iters;
    MKL_INT d = d1 * d2;
    MKL_INT d_init = d / (d_sub * d_sub);
    MKL_INT t_init = t / t_sub;
    double delta_u, delta_v;
    double lambda_tf = 0;     /* Signal To Use Heuristic As First Guess */
    double lambda_tv = .0025; /* First Guess For Constrained Problem */

    std::vector<double> z_k(t - 2, 0.0);
    std::vector<double> v_tmp(t, 0.0);

    /* Intialize Components With Power Method Iters */
    initvec(d_init, u_init, 1 / sqrt(d_init));
    regress_temporal(d_init, t_init, R_init, u_init, v_init);
    normalize(t_init, v_init);

    for (iters = 0; iters < max_iters_init; iters++) {
        /* Update Components: Regression & Normalization*/
        delta_u = update_spatial_init(d_init, t_init, R_init, u_init, v_init);
        delta_v = update_temporal_init(d_init, t_init, R_init, u_init, v_init);

        /* Check Convergence */
        if (fmax(delta_u, delta_v) < tol)
            break;
    }

    /* Upsample &/or Initialize As Needed If We Used Decimated Init */
    if (d_sub > 1 || t_sub > 1) {
        if (t_sub > 1) {
            upsample_1d(t, t_sub, v_k, v_init);
            init_dual_from_primal(t, v_k, &z_k[0]);
        } else {
            copy(t, v_init, v_k);
        }
        if (d_sub > 1) {
            initvec(d, u_k, 1 / sqrt(d));
        }
    }

    /* Loop Until Convergence Of Spatial & Temporal Components */
    for (iters = 0; iters < max_iters_main; iters++) {
        /* Update Components: Regression, Denoising, & Normalization */
        delta_u = update_spatial(R_k, u_k, v_k, &lambda_tv, pars);
        delta_v = update_temporal(d, R_k, u_k, v_k, &z_k[0], &lambda_tf, pars);

        /* Check Convergence */
        if (fmax(delta_u, delta_v) < tol) {
            /* Free Allocated Memory & Test Spatial Component Against Null */
            regress_temporal(d, t, R_k, u_k, v_k); // debias

            if (spatial_test_statistic(d1, d2, u_k) > spatial_thresh ||
                temporal_test_statistic(t, v_k) > temporal_thresh)
                return -1; // Discard Component

            return 1; // Keep Component
        }

        /* Preemptive Check To See If We're Fitting Noise */
        if (iters == 5) {
            copy(t, v_k, &v_tmp[0]);
            regress_temporal(d, t, R_k, u_k, v_k);
            if (spatial_test_statistic(d1, d2, u_k) > spatial_thresh ||
                temporal_test_statistic(t, v_k) > temporal_thresh) {
                /* Free Allocated Memory & Return */
                return -1; // Discard Component
            }
            copy(t, &v_tmp[0], v_k);
        }
    }

    /* MAXITER EXCEEDED: Free Memory & Test Spatial Component Against Null */

    regress_temporal(d, t, R_k, u_k, v_k);

    if (spatial_test_statistic(d1, d2, u_k) > spatial_thresh ||
        temporal_test_statistic(t, v_k) > temporal_thresh)
        return -1; // Discard Component

    return 1; // Keep Component
}

/* Apply TF/TV Penalized Matrix Decomposition (PMD) to factor a (d1*d2)xT
 * column major formatted video into sptial and temporal components.
 */
size_t pmd(double *R, double *R_ds, double *U, double *V, PMD_params *pars) {
    MKL_INT d1 = pars->get_bheight();
    MKL_INT d2 = pars->get_bwidth();
    MKL_INT d_sub = pars->get_d_sub();
    MKL_INT t = pars->get_t();
    MKL_INT t_sub = pars->get_t_sub();
    size_t max_components = pars->get_max_components();
    size_t consec_failures = pars->get_consec_failures();

    /* Declare & Intialize Internal Vars */
    std::vector<int> keep_flag(consec_failures, 1);
    int max_keep_flag;
    size_t i, k, good = 0;
    MKL_INT d = d1 * d2;

    /* Assign/Allocate Init Vars Based On Whether Or Not We Are Decimating */
    double *R_init, *u_init, *v_init;
    if (R_ds && (d_sub > 1 || t_sub > 1)) {
        R_init = R_ds;
        u_init = static_cast<double *>(
            malloc((d / (d_sub * d_sub)) * sizeof(double)));
        v_init = static_cast<double *>(malloc((t / t_sub) * sizeof(double)));
    } else {
        d_sub = 1;
        t_sub = 1;
        R_init = R;
        u_init = U + good * d;
        v_init = V + good * t;
    }

    /* Fill Keep Flags */
    // for (i = 0; i < consec_failures; i++) keep_flag[i] = 1;

    /* Sequentially Extract Rank 1 Updates Until We Are Fitting Noise */
    for (k = 0; k < max_components; k++, good++) {
        /* Assign Init Vars If Not Using Decimation */
        if (!(d_sub > 1 || t_sub > 1)) {
            u_init = U + good * d;
            v_init = V + good * t;
        }

        /* U[:,k] <- u_k, V[k,:] <- v_k' :
         *    min <u_k, R_k v_k> - lambda_tv ||u_k||_TV - lambda_tf ||v_k||_TF
         */

        keep_flag[k % consec_failures] = rank_one_decomposition(
            R, R_init, U + good * d, u_init, V + good * t, v_init, pars);

        /* Check Component Quality: Terminate if we're fitting noise */
        if (keep_flag[k % consec_failures] < 0) {
            max_keep_flag = -1; // current component is a failure
            for (i = 1; i < consec_failures; i++) { // check previous
                max_keep_flag = std::max(max_keep_flag,
                                         keep_flag[(k + i) % consec_failures]);
            }
            if (max_keep_flag < 0) {
                if (d_sub > 1 || t_sub > 1) {
                    free(u_init);
                    free(v_init);
                }
                return good;
            }
        }
        /* Debias Components */
        /* regress_temporal(d, t, R, U + good*d, V + good*t); Debiasing moved to
         * ROD*/

        /* Update Full Residual: R_k <- R_k - U[:,k] V[k,:] */
        cblas_dger(CblasColMajor, d, t, -1.0, U + good * d, 1, V + good * t, 1,
                   R, d);

        /* Update Downsampled Residual If Using Decimated Initialization */
        if (d_sub > 1 || t_sub > 1) {
            /* Downsample Components If Decimating Along That Dimension */
            if (d_sub > 1) {
                downsample_2d(d1, d2, d_sub, U + good * d, u_init);
            } else {
                copy(d, U + good * d, u_init);
            }
            if (t_sub > 1) {
                downsample_1d(t, t_sub, V + good * t, v_init);
            } else {
                copy(t, V + good * t, v_init);
            }

            /* Update Downsampled Residual: R_ds <- R_ds - u_ds v_ds' */
            cblas_dger(CblasColMajor, d / (d_sub * d_sub), t / t_sub, -1.0,
                       u_init, 1, v_init, 1, R_init, d / (d_sub * d_sub));
        }

        /* Make Sure We Overwrite Failed Components */
        if (keep_flag[k % consec_failures] < 0) {
            good--;
        }
    }

    /* MAXCOMPONENTS EXCEEDED: Terminate Early */
    if (d_sub > 1 || t_sub > 1) {
        free(u_init);
        free(v_init);
    }
    return good;
}

int get_block(int row_start, int row_end, int col_start, int col_end,
              int frame_start, int frame_end, void *requester, double *buffer,
              size_t n) {

    int res;
    int i = 0;
    int positions[6];
    positions[i++] = row_start;
    positions[i++] = row_end;
    positions[i++] = col_start;
    positions[i++] = col_end;
    positions[i++] = frame_start;
    positions[i] = frame_end;

    res = zmq_send(requester, positions, sizeof(positions), 0);
    if (res != sizeof(positions)) {
        perror("zmq_send() failed");
        exit(1);
    }
    res = zmq_recv(requester, buffer, n * sizeof(double), 0);
    if (res == -1) {
        perror("zmq_recv() failed");
        exit(1);
    }

    return 1;
}

/* Wrap TV/TF Penalized Matrix Decomposition with OMP directives to enable
 * parallel, block-wiseprocessing of large datasets in shared memory.
 */
void batch_pmd(double **Up, double **Vp, size_t *K, const int num_blocks,
               PMD_params *pars, size_t *indices) {
    // Create FFT Handle So It can Be Shared Across Threads
    DFTI_DESCRIPTOR_HANDLE FFT;
    MKL_LONG status;
    MKL_LONG L = 256; /* Size Of Subsamples in welch estimator */
    status = DftiCreateDescriptor(&FFT, DFTI_DOUBLE, DFTI_REAL, 1, L);
    status = DftiSetValue(FFT, DFTI_PACKED_FORMAT, DFTI_PACK_FORMAT);
    status = DftiCommitDescriptor(FFT);
    if (status != 0)
        std::cerr << "Error while creating MKL_FFT Handle: " << status
                  << std::endl;

    // Loop Over All Patches In Parallel
    pars->set_FFT(static_cast<void *>(&FFT));

    int bwidth = pars->get_bwidth();
    int bheight = pars->get_bheight();
    int num_frames = pars->get_t(); // frames
    int d_sub = pars->get_d_sub();
    int t_sub = pars->get_t_sub();

    void *context = zmq_ctx_new();
    /*
    int io_threads = NUM_THREADS;
    zmq_ctx_set(context, ZMQ_IO_THREADS, io_threads);
    if (zmq_ctx_get(context, ZMQ_IO_THREADS) != io_threads) {
        fprintf(stderr, "io_threads not set properly");
        exit(1);
    }
    */

#pragma omp parallel shared(FFT)
    {
        // residual for a block
        std::vector<double> Rp_block((bheight * bwidth * num_frames));

        // residual downsampled for a block
        std::vector<double> Rp_ds_block((bheight / d_sub) * (bwidth / d_sub) *
                                        (num_frames / t_sub));

        void *requester = zmq_socket(context, ZMQ_REQ);
        int rc = zmq_connect(requester, "ipc://trefide.ipc");
        if (rc != 0) {
            perror("zmq_connect() failed");
            exit(1);
        }

#pragma omp for schedule(guided)
        for (int block = 0; block < num_blocks; block++) {
            size_t block_row = indices[2 * block];
            size_t block_col = indices[2 * block + 1];

            int row_start = block_row;
            int row_end = row_start + bheight;
            int col_start = block_col;
            int col_end = col_start + bwidth;
            int frame_start = 0;
            int frame_end = num_frames;
            get_block(row_start, row_end, col_start, col_end, frame_start,
                      frame_end, requester, &Rp_block[0],
                      (bheight * bwidth * num_frames));

            // Downsample block residual
            if (d_sub > 1 || t_sub > 1) {
                downsample_3d(bheight, bwidth, d_sub, num_frames, t_sub,
                              Rp_block.data(), Rp_ds_block.data());
            }

            K[block] = pmd(Rp_block.data(), Rp_ds_block.data(), Up[block],
                           Vp[block], pars);
        }

        zmq_close(requester);
    }

    zmq_ctx_destroy(context);

    // Free MKL FFT Handle
    status = DftiFreeDescriptor(&FFT);
    if (status != 0) {
        std::cerr << "Error while deallocating MKL_FFT Handle: " << status
                  << std::endl;
    }
}
