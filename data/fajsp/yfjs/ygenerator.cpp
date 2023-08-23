/////////////////////////////////////////////////////////////////////////
// 
// First random instance generator described in 
// 
// E.G. Birgin, P. Feofiloff, C.G. Fernandes, E.L. de Melo, M.T.I. Oshiro, 
// D.P. Ronconi, A MILP model for an extended version of the Flexible Job 
// Shop Problem, submitted, 2012,
// 
// that produces instances consisting of Y-jobs. 
// 
/////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <set>
#include <math.h>

using namespace std;

#define MINPT 20  /* minimum processing time */
#define MAXPT 200 /* maximum processing time */

void setInitSeed(int);
double rndD();
int rndInt(int, int);
void fname(char*, char*, int, int);

int X = 1;

/* Set the initial seed for the pseudorandom numbers */
void setInitSeed(int seed) {
  if (seed > 0) {
    X = seed;
  } else {
    printf("The initial seed must be greater than 0!\n");
    X += 2147483647;    
  }
}

/* Returns a pseudorandom number p such that 0 < p < 1.
 * This procedure was proposed by Bratley, Fox and Schrage */ 
double rndD() {
  const int a = 16807;
  const int b = 127773;
  const int c = 2836;
  const int m = 2147483647;
  int k;
  
  k = X / b;
  X = a * (X % b) - k * c;
  if (X < 0)
    X += m;
  return (double)X / (double)m;
}

/* Return a pseudorandom integer number between a and b. */
int rndInt(int a, int b) {
  long long ba1 = b - a + 1;
  long long aa = a;
  int r = (int)floor(rndD() * ba1 + aa);

  if (r <= 0)
    printf("Maybe there was a problem with the "
           "pseudorandom generator.\n");  
  return r;
}

void fname(char* filename, char *prefix, int t, int insts) {
  if (insts == 1) {
    sprintf(filename, "%s", prefix);
  } else if (insts < 10) {
    sprintf(filename, "%s%d", prefix, t);
  } else if (insts < 100) {
    sprintf(filename, "%s%02d", prefix, t);
  } else {
    sprintf(filename, "%s%03d", prefix, t);
  }
}

int main () {
  int seed;
  int njob, nmach, nop, opj, flex, arcs, insts;
  int i, j, k, t;
  char prefix[101], filename[111];;
  FILE* outfile;
  set<int> usedSeeds;

  printf("Number of jobs: ");
  scanf("%d", &njob);
  printf("Number of operations for each job: ");
  scanf("%d", &opj);
  nop = njob * opj;
  printf("Number of machines: ");
  scanf("%d", &nmach);
  printf("Number of possible machines per operation: ");
  scanf("%d", &flex);
  printf("Instance file name (max. 100 chars): ");
  scanf("%s", prefix);
  printf("Number of instances to create: ");
  scanf("%d", &insts);

  /* Matrix of the processing times.                        */
  int **prtime = (int**) malloc(nmach * sizeof(int*));
  for (j = 0; j < nmach; j++) {
    prtime[j] = (int*) malloc (nop * sizeof(int));
  }

  seed = time(NULL);
  setInitSeed(seed);
  usedSeeds.insert(seed);
  for (t = 1; t <= insts; t++) {

    /* Create the dag. All jobs are paths of the same size */
    vector< vector<int> > dag(nop);
    arcs = 0;
    for (i = 0; i < njob; i++) {
      int u, v;
      u = rndInt(1, opj) - 1;
      v = rndInt(1, opj) - 1;
      if (u == v) {
	for (j = 1; j < opj; j++) {
	  k = i*opj + j;
	  dag[k-1].push_back(k);
	  arcs++;
	}
      } else {
	if (u > v) {
	  k = u;
	  u = v;
	  v = k;
	}
	for (j = 1; j < opj; j++) {
	  if (j == v) continue;
	  k = i*opj + j;
	  dag[k-1].push_back(k);
	  arcs++;
	}
	dag[i*opj + u].push_back(i*opj + v);
	arcs++;
      }
    }
    
    /* Each set contains the possible machines for the operation */
    vector< set<int> >possmachs(nop);
    for (i = 0; i < nop; i++) {
      for (j = 0; j < flex; j++) {
	possmachs[i].insert(rndInt(1, nmach) - 1);
      }
    }
    
    for (j = 0; j < nmach; j++) {
      for (i = 0; i < nop; i++) {
	if (possmachs[i].find(j) != possmachs[i].end()) {
	  prtime[j][i] = rndInt(MINPT, MAXPT);
	}
      }
    }
    
    fname(filename, prefix, t, insts);
    outfile = fopen(filename, "w");
    fprintf(outfile, "# Number of jobs: %d\n", njob);
    fprintf(outfile, "# Number of operations for each job: %d\n", opj);
    fprintf(outfile, "# Number of machines: %d\n", nmach);
    fprintf(outfile, 
	    "# Number of possible machines per operation: %d\n", flex);
    
    fprintf(outfile, "%d %d %d\n", nop, arcs, nmach);
    for (i = 0; i < nop; i++) {
      for (j = 0; j < dag[i].size(); j++) {
	fprintf(outfile, "%d %d\n", i, dag[i][j]);
      }
    }
    
    set<int>::iterator setIt;
    for (i = 0; i < nop; i++) {
      fprintf(outfile, "%d", possmachs[i].size());
      for (setIt = possmachs[i].begin(); setIt != possmachs[i].end();
	   setIt++) {
	j = *setIt;
	fprintf(outfile, " %d %d", j, prtime[j][i]);
      }
      fprintf(outfile, "\n");
    }
    
    fclose(outfile);

    seed = rndInt(1, 2147483647);
    while (usedSeeds.find(seed) != usedSeeds.end()) {
      seed = rndInt(1, 2147483647);
    }
    setInitSeed(seed);
    usedSeeds.insert(seed);
  }
    
  for (j = 0; j < nmach; j++) {
    free(prtime[j]);
  }
  free(prtime);
  return 0;
}
