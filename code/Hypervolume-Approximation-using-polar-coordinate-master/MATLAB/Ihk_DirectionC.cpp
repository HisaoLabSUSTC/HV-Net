/*
	Ihk_DirectionC.cpp		Mex C++ file for I_h^k hypervolume indicator approximation by uniform directions
	
	Usage: Ihk_DirectionC(points, reference point, param_k, samples, sphere)

	points: A row-wise set of population
	reference point: A row vector for hypervolume, please make sure it is dominated by any point in points
	param_k: parameter k in I_h^k
	samples: A row-wise set of inverted uniformly distributed directions(You need to generate samples in MATLAB by appropriate method)
	sphere: Surface area of m-ball in R^m_+ (Easy to calculate in MATLAB)

	Editor: Jingda Deng
	Email: jingddeng2-c@my.cityu.edu.hk
	Date: 30 Dec, 2017
*/

#include "mex.h"
#include <math.h>
#include <functional>
#include <algorithm>
#include <numeric>

template <typename Container>
struct compare_indirect_index_descend
  {
  const Container& container;
  compare_indirect_index_descend( const Container& container ): container( container ) { }
  bool operator () ( size_t lindex, size_t rindex ) const
    {
    return container[ lindex ] > container[ rindex ];
    }
  };

void Index_Sort(double *x, int *idx, int m)
{	
	std::iota(idx, idx+m, 0);
	std::sort( idx, idx+m, compare_indirect_index_descend <decltype(x)> ( x ) );		// referred to http://www.cplusplus.com/forum/beginner/116101/
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	
	int m, n, w, param_k;
	double *points, *reference, *samples, sphere, v, sum;
	bool flag;
	double *hv;
	points = mxGetPr(prhs[0]);
	reference = mxGetPr(prhs[1]);
	param_k = *mxGetPr(prhs[2]);
	samples = mxGetPr(prhs[3]);
	sphere = *mxGetPr(prhs[4]);
	
	n = mxGetM(prhs[0]);
	m = mxGetN(prhs[0]);	
	w = mxGetM(prhs[3]);
	
	plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
	hv = mxGetPr(plhs[0]);
	
	double *alpha = new double[param_k];
	for(int i=0; i<param_k; i++)
	{
		alpha[i] = 1./(i+1);
		for(int j=1; j<=i; j++)
		{
			alpha[i] *= (double)(param_k - j)/(double)(n - j);
		}
	}	
	double *Length = new double[n+1];
	int *Index = new int[n+1];
	Length[n] = 0.;
	Index[n] = n;
	
	for(int i=0; i<w; i++)
	{
		for(int j=0; j<n; j++)
		{	
			Length[j] = 1e30;	
			Index[j] = j;
			for(int k=0; k<m; k++)
			{
				v = -samples[k*w+i]*(points[k*n+j] - reference[k]);
				if(v<Length[j])
					Length[j] = v;
			}			
			Length[j] = pow(Length[j], m);
		}
		Index_Sort(Length, Index, n);
		sum = 0.;
		for(int j=param_k-1; j>=0; j--)
		{
			sum += (Length[Index[j]] - Length[Index[j+1]]) * alpha[j];
			*(hv+Index[j]) = *(hv+Index[j]) + sum;
		}
	}
	
	for(int i=0; i<n; i++)
		*(hv+i) = *(hv+i)/m/w*sphere;
	
	delete[] alpha;
	delete[] Length;
	delete[] Index;
}
