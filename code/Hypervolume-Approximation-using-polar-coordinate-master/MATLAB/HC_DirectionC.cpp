/*
	HC_DirectionC.cpp		Mex C++ file for hypervolume contribution approximation by uniform directions
	
	Usage: HC_DirectionC(points, reference point, samples, sphere)

	points: A row-wise set of population
	reference point: A row vector for hypervolume, please make sure it is dominated by any point in points
	samples: A row-wise set of inverted uniformly directions(You need to generate samples in MATLAB in different methods)
	sphere: Surface area of m-ball in R^m_+

	Editor: Jingda Deng
	Email: jingddeng2-c@my.cityu.edu.hk
	Date: 30 Dec, 2017
*/

#include "mex.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	
	int m, n, w, ID;
	double *points, *reference, *samples, sphere, Length, Length2;
	bool flag;
	double *hv;
	points = mxGetPr(prhs[0]);
	reference = mxGetPr(prhs[1]);
	samples = mxGetPr(prhs[2]);
	sphere = *mxGetPr(prhs[3]);
	
	n = mxGetM(prhs[0]);
	m = mxGetN(prhs[0]);	
	w = mxGetM(prhs[2]);
	
	plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
	hv = mxGetPr(plhs[0]);
	
	double ming, v;
	
	for(int i=0; i<w; i++)
	{
		Length = 0.;
		for(int j=0; j<n; j++)
		{
			ming = 1e30;	
			for(int k=0; k<m; k++)
			{
				v = -samples[k*w+i]*(points[k*n+j] - reference[k]);
				if(v<ming)
					ming = v;
			}
			if(ming>Length)
			{
				ID = j;
				Length2 = Length;
				Length = ming;
			}
			else
			{
				if(ming>Length2)
					Length2 = ming;
			}
		}
		*(hv+ID) += pow(Length, m) - pow(Length2, m);
	}
	
	for(int i=0; i<n; i++)
		*(hv+i) = *(hv+i)/m/w*sphere;
}
