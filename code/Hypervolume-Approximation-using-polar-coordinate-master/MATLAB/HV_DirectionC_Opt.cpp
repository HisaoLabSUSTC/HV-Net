/*
	HV_DirectionC.cpp		Mex C++ file for hypervolume approximation by uniform directions
	
	Usage: HV_DirectionC(points, reference point, samples, sphere)

	points: A row-wise set of population
	reference point: A row vector for hypervolume, please make sure it is dominated by any point in points
	samples: A row-wise set of random normalied directions(You need to generate samples in MATLAB with "rand" in advance)
	sphere: Surface area of m-ball in R^m_+

	Editor: Jingda Deng
	Email: jingddeng2-c@my.cityu.edu.hk
	Date: 30 Dec, 2017
*/

#include "mex.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	
	int m, n, w;
	double *points, *reference, *samples, sphere, Length;
	double *hv;
	points = mxGetPr(prhs[0]);
	reference = mxGetPr(prhs[1]);
	samples = mxGetPr(prhs[2]);
	sphere = *mxGetPr(prhs[3]);
	plhs[0] = mxCreateDoubleScalar(0.0);
	hv = mxGetPr(plhs[0]);
		
	n = mxGetM(prhs[0]);
	m = mxGetN(prhs[0]);	
	w = mxGetM(prhs[2]);
	
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
				if(v<Length)
				{
					ming = 0;
					break;
				}
				else
				{
					if(v<ming)
						ming = v;
				}
			}
			if(ming>Length)
				Length = ming;
		}
		//mexPrintf("%f \n", Length);
		*hv += pow(Length, m);
	}
	*hv = *hv/m/w*sphere;
}
