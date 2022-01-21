/*
	HC_PolarC.cpp		Mex C++ file for hypervolume contribution approximation by polar coordinate
	
	Usage: HC_PolarC(points, reference point, samples)

	points: A row-wise set of population
	reference point: A row vector for hypervolume, please make sure it is dominated by any point in points
	samples: number of samples
	
	Editor: Jingda Deng
	Email: jingddeng2-c@my.cityu.edu.hk
	Date: 30 Dec, 2017
*/

#include "mex.h"
#include <cmath>
#define PI  3.1415926535897932384626433832795

double Jaccobi(double* angle, int m)
{
	double p = 1;
	for(int i=0; i<m; i++)
	{
		p = p*pow(sin(angle[i]), m-i-1);
	}
	return p;
}

void ConstructLattice(int m, int w, double **weights, double *jacobians)
{
	// number-theoretic points
	double eps = 1e-5;
	int p = (m-1)*2+1;		// the dimension of angles is 1 smaller than m
	bool flag = true;
	while(flag)
	{
		flag = false;
		for(int i=3; i<=floor(sqrt(p)); i++)
		{
			if(p%i==0)
			{
				flag = true;
				break;
			}
		}
		if(flag)
			p = p+2;
	}
	int *a = new int[m-1];	// the dimension of angles is 1 smaller than m
	a[0] = 1;
	for(int i=1; i<m-1; i++)
	{
		double temp = 2*cos(2*PI*i/(double)p);
		a[i] = int(w*(fabs(temp) - floor(fabs(temp)))+0.5);
	}

	double *angle = new double[m-1];
	for(int i=0; i<w; i++)
	{
		// lattice points and jacobians
		for(int j=0; j<m-1; j++)
		{
			double temp = (i+1)*a[j]/(double)w;
			angle[j] = temp - floor(temp);
			if(angle[j]<eps)
				angle[j] = angle[j]+eps;
			if(angle[j]>1-eps)
				angle[j] = angle[j]-eps;
			angle[j] = angle[j]*PI/2.0;
		}
		jacobians[i] = Jaccobi(angle, m-1);
		
		// construct weights and the inverse
		double sum = 0.;
		double sum2 = 0.;
		for(int j=0; j<m; j++)
		{
			weights[i][j] = 1;
			for(int k=0; k<m-j-1; k++)
				weights[i][j] = weights[i][j]*sin(angle[k]);
			if(j>0)
				weights[i][j] = weights[i][j]*cos(angle[m-j-1]);
			weights[i][j] = 1.0/weights[i][j];
		}
	}
	
	delete[] angle;
	delete[] a;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	
	int m, n, w, ID;
	double *points, *reference, Length, Length2;
	bool flag;
	double *hv;
	points = mxGetPr(prhs[0]);
	reference = mxGetPr(prhs[1]);
	
		
	n = mxGetM(prhs[0]);
	m = mxGetN(prhs[0]);	
	w = *mxGetPr(prhs[2]);
	
	plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
	hv = mxGetPr(plhs[0]);
	
	double *jacobians = new double[w];
	double **weights = new double*[w];
	for(int i=0; i<w; i++)
	{
		weights[i] = new double[m];
	}
	ConstructLattice(m, w, weights, jacobians);
		
	double ming, v;
	for(int i=0; i<w; i++)
	{
		Length = 0.;
		Length2 = 0.;
		ID = 0;
		for(int j=0; j<n; j++)
		{
			ming = 1e30;	
			for(int k=0; k<m; k++)
			{
				v = -weights[i][k]*(points[k*n+j] - reference[k]);
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
		*(hv+ID) += (pow(Length, m) - pow(Length2, m))*jacobians[i];
	}
	for(int i=0; i<n; i++)
		*(hv+i) = *(hv+i)/m/w*pow(PI/2, m-1);
	
	for(int i=0; i<w; i++)
	{
		delete[] weights[i];
	}
	delete[] weights;
	delete[] jacobians;
}
