/////////////////////////////////////////////////////////////////////////
// 
// Second random instance generator described in 
// 
// E.G. Birgin, P. Feofiloff, C.G. Fernandes, E.L. de Melo, M.T.I. Oshiro, 
// D.P. Ronconi, A MILP model for an extended version of the Flexible Job 
// Shop Problem, submitted, 2012.
// 
/////////////////////////////////////////////////////////////////////////

/*
INSTRUCTIONS:

The number of jobs (n) and machines (m) are defined in column 0 and 1 
of LisPar. Column 2 defines how many instances nxm must be generated. 
Var NumGru defines how many lines of LisPar must be used.
Example:
   NumGru=2;
   LisPar[0][0]=4; LisPar[0][1]=5;  LisPar[0][2]=2;
   LisPar[1][0]=4; LisPar[1][1]=10; LisPar[1][2]=2;
This code generates 2 instances 4x5 and 2 instances 4x10.

*/

#include <stdio.h>
#include <stdlib.h>

FILE *ArqSai;
long int X=2010;
long int CopX;
int Cont, NumJob, NumMaq, ComprJob, NumOPJ, TipoJob, RamifJob, NumMA, 
  NumMedMA, NumMinMA, NumMaxMA, NumOpe, **MatMaqAlt, **MatTemPro, 
  **MatOpeJob, NumGru, NumIns, NumArc;
float Flex, f, g, Beta;
char NomSai[102], LisPar[1000][3], NomSai2[102];

float AuxGerNumAleTai(){
  long int k, a=16807, b=127773, c=2836, m=pow(2,31)-1;
  float Val, X_float, m_float;
  
  k = X/b;
  X = a*(X%b)-k*c;
  if (X < 0){
    X = X+m;
  }
  X_float = X;
  m_float = m;
  Val = X_float/m_float;
  return Val;
}

int GerNumAleTai(int a, int b){
  int Val;

  Val = a+(float)(AuxGerNumAleTai()*(b-a+1));
  return V al;
} 

void GerIns(){
  int i, j, k, l, a, b, SeqDesm, SeqMont;

  NumJob = LisPar[Cont][0];
  NumMaq = LisPar[Cont][1];
  f = Flex*NumMaq;
  NumMedMA = (int)(Flex*NumMaq);
  if (f > NumMedMA){
    NumMedMA = NumMedMA+1;
  }
  a = NumMedMA-1;
  if (a > NumMaq-NumMedMA){
    a = NumMaq-NumMedMA;
  }
  if (a > NumMaq*0.2){
    a = (int)(NumMaq*0.2);
  }
  NumMinMA = NumMedMA-a;
  NumMaxMA = NumMedMA+a;
  MatMaqAlt = malloc((NumJob*NumMaq*3)*sizeof(int));
  if (MatMaqAlt == NULL) {
    printf("\n\n Memory allocation error.\n\n");
    exit(0); 
  }
  for (i=0; i<NumJob*NumMaq*3; i++){
    MatMaqAlt[i] = malloc((NumMaq+1)*sizeof(int));
    if (MatMaqAlt[i] == NULL) {
      printf("\n\n Memory allocation error.\n\n");
      exit(0); 
    } 
  } 
  MatOpeJob = malloc((NumJob)*sizeof(int));
  if (MatOpeJob == NULL) {
    printf("\n\n Memory allocation error.\n\n");
    exit(0);
  }
  for (i=0; i<NumJob; i++){
    MatOpeJob[i] = malloc((NumMaq*3+6)*sizeof(int)); 
    // From position NumMaq*3 on, it stores: NOJ; TipoJob; RamifJob; SeqDesm; SeqMont.
    if (MatOpeJob[i] == NULL) {
      printf("\n\n Memory allocation error.\n\n");
      exit(0);
    }
  }
  MatTemPro = malloc((NumJob*NumMaq*3)*sizeof(int));
  if (MatTemPro == NULL) {
    printf("\n\n Memory allocation error.\n\n");
    exit(0);
  }
  for (i=0; i<NumJob*NumMaq*3; i++){
    MatTemPro[i] = malloc((NumMaq)*sizeof(int));
    if (MatTemPro[i] == NULL) {
      printf("\n\n Memory allocation error.\n\n");
      exit(0);
    }
  }
  for (i=0; i<NumJob*NumMaq*3; i++){
    for (j=0; j<NumMaq+1; j++){
      MatMaqAlt[i][j] = 0;
    }
  }
  for (i=0; i<NumJob; i++){
    for (j=0; j<NumMaq*3+1; j++){
      MatOpeJob[i][j] = 0;
    }
  }
  for (i=0; i<NumJob*NumMaq*3; i++){
    for (j=0; j<NumMaq; j++){
      MatTemPro[i][j] = 0;
    }
  }
  NumOpe = NumArc = 0;
  for (i=0; i<NumJob; i++){
    TipoJob = GerNumAleTai(1,3);
    MatOpeJob[i][NumMaq*3+1] = TipoJob;
    RamifJob = GerNumAleTai(2,3);
    MatOpeJob[i][NumMaq*3+2] = RamifJob;
    a = NumMaq/2;
    if (NumMaq%2 > 0){
      a = a+1;
    }
    ComprJob = GerNumAleTai(a,NumMaq);
    MatOpeJob[i][NumMaq*3+5] = ComprJob;
    if (TipoJob == 1){
      SeqMont = GerNumAleTai(2,ComprJob);
      NumOPJ = ComprJob+((SeqMont-1)*(RamifJob-1));
      MatOpeJob[i][NumMaq*3+4] = SeqMont;
      NumArc = NumArc+NumOPJ-1;
    }
    else if (TipoJob == 2){
      SeqDesm = GerNumAleTai(1,ComprJob-1);
      NumOPJ = ComprJob+((ComprJob-SeqDesm)*(RamifJob-1));
      MatOpeJob[i][NumMaq*3+3] = SeqDesm;
      NumArc = NumArc+NumOPJ-1;
    }
    else if (TipoJob == 3){
      SeqDesm = GerNumAleTai(1,ComprJob-2);
      SeqMont = GerNumAleTai(SeqDesm+2,ComprJob);
      NumOPJ = ComprJob+(SeqMont-SeqDesm-1)*(RamifJob-1);
      MatOpeJob[i][NumMaq*3+3] = SeqDesm;
      MatOpeJob[i][NumMaq*3+4] = SeqMont;
      NumArc = NumArc+NumOPJ-2+RamifJob;
    }
    MatOpeJob[i][NumMaq*3] = NumOPJ;
    for (j=0; j<NumOPJ; j++){
      NumMA = GerNumAleTai(NumMinMA,NumMaxMA);
      MatMaqAlt[NumOpe][NumMaq] = NumMA;
      for (k=0; k<NumMA; k++){
	MatMaqAlt[NumOpe][k] = GerNumAleTai(1,NumMaq);
	a = 0;
	for (l=0; l<k; l++){
	  if (MatMaqAlt[NumOpe][k] == MatMaqAlt[NumOpe][l]){
	    a = 1;
	  }
	}
	if (a == 1){
	  while (a == 1){
	    a = 0;
	    MatMaqAlt[NumOpe][k] = GerNumAleTai(1,NumMaq);
	    for (l=0; l<k; l++){
	      if (MatMaqAlt[NumOpe][k] == MatMaqAlt[NumOpe][l]){
		a = 1;
	      }
	    }
	  }
	}
      }
      for (k=0; k<NumMA; k++){
	if (k == 0){
	  MatTemPro[NumOpe][MatMaqAlt[NumOpe][k]-1] = GerNumAleTai(1,99);
	}
	else{
	  if (3*(MatTemPro[NumOpe][MatMaqAlt[NumOpe][0]-1]) <= 99){
	    MatTemPro[NumOpe][MatMaqAlt[NumOpe][k]-1] = 
	      GerNumAleTai(MatTemPro[NumOpe][MatMaqAlt[NumOpe][0]-1],
			   3*MatTemPro[NumOpe][MatMaqAlt[NumOpe][0]-1]);
	  }
	  else{
	    MatTemPro[NumOpe][MatMaqAlt[NumOpe][k]-1] = 
	      GerNumAleTai(MatTemPro[NumOpe][MatMaqAlt[NumOpe][0]-1],99);
	  }
	}
      }
      MatOpeJob[i][j] = NumOpe+1;
      NumOpe = NumOpe+1;
    } 
  }
  sprintf(NomSai, "%s%d", NomSai2, NumIns+1);
  ArqSai=fopen(NomSai, "w");
  fprintf(ArqSai, "%d ", NumOpe);
  fprintf(ArqSai, "%d ", NumArc);
  fprintf(ArqSai, "%d", NumMaq);
  for (i=0; i<NumJob; i++){
    a = 0;
    NumOPJ = MatOpeJob[i][NumMaq*3];
    TipoJob = MatOpeJob[i][NumMaq*3+1];
    RamifJob = MatOpeJob[i][NumMaq*3+2];
    SeqDesm = MatOpeJob[i][NumMaq*3+3];
    SeqMont = MatOpeJob[i][NumMaq*3+4];
    ComprJob = MatOpeJob[i][NumMaq*3+5];
    if (TipoJob == 1){
      for (j=0; j<RamifJob; j++){
	for (k=0; k<SeqMont-2; k++){
	  fprintf(ArqSai, "\n%d %d", MatOpeJob[i][a]-1, MatOpeJob[i][a+1]-1);
	  a = a+1;
	}
	fprintf(ArqSai, "\n%d %d", MatOpeJob[i][a]-1, 
		MatOpeJob[i][RamifJob*(SeqMont-1)]-1);
	a = a+1;
      }
      for (j=(RamifJob*(SeqMont-1)); j<NumOPJ-1; j++){
	fprintf(ArqSai, "\n%d %d", MatOpeJob[i][j]-1, MatOpeJob[i][j+1]-1);
      }
    }
    if (TipoJob == 2){
      for (j=0; j<(SeqDesm-1); j++){
	fprintf(ArqSai, "\n%d %d", MatOpeJob[i][j]-1, MatOpeJob[i][j+1]-1);
      }
      a = SeqDesm;
      for (j=0; j<RamifJob; j++){
	fprintf(ArqSai, "\n%d %d", MatOpeJob[i][(SeqDesm-1)]-1, MatOpeJob[i][a]-1);
	for (k=0; k<ComprJob-SeqDesm-1; k++){
	  fprintf(ArqSai, "\n%d %d", MatOpeJob[i][a]-1, MatOpeJob[i][a+1]-1);
	  a = a+1;
	}
	a = a+1;
      }
    }
    if (TipoJob == 3){
      for (j=0; j<(SeqDesm-1); j++){
	fprintf(ArqSai, "\n%d %d", MatOpeJob[i][j]-1, MatOpeJob[i][j+1]-1);
      }
      a = SeqDesm;
      for (j=0; j<RamifJob; j++){
	fprintf(ArqSai, "\n%d %d", MatOpeJob[i][(SeqDesm-1)]-1,MatOpeJob[i][a]-1);
	for (k=0; k<SeqMont-SeqDesm-2; k++){
	  fprintf(ArqSai, "\n%d %d", MatOpeJob[i][a]-1, MatOpeJob[i][a+1]-1);
	  a = a+1;
	}
	fprintf(ArqSai, "\n%d %d", MatOpeJob[i][a]-1, 
		MatOpeJob[i][SeqDesm+(RamifJob*(SeqMont-SeqDesm-1))]-1);
	a = a+1;
      }
      for (j=(SeqDesm+(RamifJob*(SeqMont-SeqDesm-1))); j<NumOPJ-1; j++){
	fprintf(ArqSai, "\n%d %d", MatOpeJob[i][j]-1, MatOpeJob[i][j+1]-1);
      }
    }
  }
  
  for (i=0; i<NumOpe; i++){
    fprintf(ArqSai, "\n%d", MatMaqAlt[i][NumMaq]);
    for (j=0; j<MatMaqAlt[i][NumMaq]; j++){
      fprintf(ArqSai, " %d %d", MatMaqAlt[i][j]-1, MatTemPro[i][MatMaqAlt[i][j]-1]);
    }
  }
  
  fprintf(ArqSai,"\n");
  for (i=0; i<NumJob; i++){
    NumOPJ = MatOpeJob[i][NumMaq*3];
    TipoJob = MatOpeJob[i][NumMaq*3+1];
    RamifJob = MatOpeJob[i][NumMaq*3+2];
    SeqDesm = MatOpeJob[i][NumMaq*3+3];
    SeqMont = MatOpeJob[i][NumMaq*3+4];
    ComprJob = MatOpeJob[i][NumMaq*3+5];
    fprintf(ArqSai, "\n\nJob %d:\n", i);
    a = 0;
    if (TipoJob == 1){
      for (k=0; k<SeqMont-1; k++){
	fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	if (k < SeqMont-2){
	  fprintf(ArqSai, "-");
	}
	a = a+1;
      }
      fprintf(ArqSai, "\n");
      for (j=0; j<SeqMont-2; j++){
	fprintf(ArqSai, "     ");
      }
      fprintf(ArqSai, "    \\ \n");
      if (RamifJob == 2){
	for (j=0; j<SeqMont-1; j++){
	  fprintf(ArqSai, "     ");
	}
	for (j=(RamifJob*(SeqMont-1)); j<NumOPJ; j++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][j]-1);
	  if (j < NumOPJ-1){
	    fprintf(ArqSai, "-");
	  }
	}
	fprintf(ArqSai, "\n");
	for (j=0; j<SeqMont-2; j++){
	  fprintf(ArqSai, "     ");
	}
	fprintf(ArqSai, "    /\n");
	for (k=0; k<SeqMont-1; k++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	  if (k < SeqMont-2){
	    fprintf(ArqSai, "-");
	  }
	  a = a+1;
	}
      }
      else{
	for (k=0; k<SeqMont-1; k++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	  fprintf(ArqSai, "-");
	  a = a+1;
	}
	for (j=(RamifJob*(SeqMont-1)); j<NumOPJ; j++){
	  fprintf(ArqSai,"%3d ",MatOpeJob[i][j]-1);
	  if (j < NumOPJ-1){
	    fprintf(ArqSai, "-");
	  }
	}
	fprintf(ArqSai, "\n");
	for (j=0; j<SeqMont-2; j++){
	  fprintf(ArqSai, "     ");
	}
	fprintf(ArqSai, "    /\n");
	for (k=0; k<SeqMont-1; k++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	  if (k < SeqMont-2){
	    fprintf(ArqSai, "-");
	  }
	  a = a+1;
	}
      }
    }
    else if (TipoJob == 2){
      for (j=0; j<SeqDesm; j++){
	fprintf(ArqSai, "     ");
      }
      a = SeqDesm;
      for (k=0; k<ComprJob-SeqDesm; k++){
	fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	if (k < ComprJob-SeqDesm-1){
	  fprintf(ArqSai, "-");
	}
	a = a+1;
      }
      fprintf(ArqSai, "\n");
      for (j=0; j<SeqDesm-1; j++){
	fprintf(ArqSai, "     ");
      }
      fprintf(ArqSai, "    / \n");
      
      for (k=0; k<SeqDesm; k++){
	fprintf(ArqSai, "%3d ", MatOpeJob[i][k]-1);
	if (k < SeqDesm-1){
	  fprintf(ArqSai, "-");
	}
      }
      if (RamifJob == 2){
	fprintf(ArqSai, "\n");
	for (j=0; j<SeqDesm-1; j++){
	  fprintf(ArqSai, "     ");
	}
	fprintf(ArqSai, "    \\ \n");
	for (j=0; j<SeqDesm; j++){
	  fprintf(ArqSai, "     ");
	}
	for (k=0; k<ComprJob-SeqDesm; k++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	  if (k < ComprJob-SeqDesm-1){
	    fprintf(ArqSai, "-");
	  }
	  a = a+1;
	}
      }
      else{
	fprintf(ArqSai, "-");
	for (k=0; k<ComprJob-SeqDesm; k++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	  if (k < ComprJob-SeqDesm-1){
	    fprintf(ArqSai, "-");
	  }
	  a = a+1;
	}
	fprintf(ArqSai, "\n");
	for (j=0; j<SeqDesm-1; j++){
	  fprintf(ArqSai, "     ");
	}
	fprintf(ArqSai, "    \\ \n");
	for (j=0; j<SeqDesm; j++){
	  fprintf(ArqSai, "     ");
	}
	for (k=0; k<ComprJob-SeqDesm; k++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	  if (k < ComprJob-SeqDesm-1){
	    fprintf(ArqSai, "-");
	  }
	  a = a+1;
	}
      }
    }
    else if (TipoJob == 3){
      for (j=0; j<SeqDesm; j++){
	fprintf(ArqSai, "     ");
      }
      a = SeqDesm;
      for (k=0; k<SeqMont-SeqDesm-1; k++){
	fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	if (k < SeqMont-SeqDesm-2){
	  fprintf(ArqSai, "-");
	}
	a = a+1;
      }
      fprintf(ArqSai, "\n");
      for (j=0; j<SeqDesm-1; j++){
	fprintf(ArqSai, "     ");
      }
      fprintf(ArqSai, "    / ");
      for (k=0; k<SeqMont-SeqDesm-2; k++){
	fprintf(ArqSai, "     ");
      }
      fprintf(ArqSai, "   \\ \n");
      for (k=0; k<SeqDesm; k++){
	fprintf(ArqSai, "%3d ", MatOpeJob[i][k]-1);
	if (k < SeqDesm-1){
	  fprintf(ArqSai, "-");
	}
      }
      if (RamifJob == 2){
	for (j=SeqDesm; j<SeqMont-1; j++){
	  fprintf(ArqSai, "     ");
	}
	fprintf(ArqSai, " ");
	for (j=(SeqDesm+(RamifJob*(SeqMont-SeqDesm-1))); j<NumOPJ; j++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][j]-1);
	  if (j < NumOPJ-1){
	    fprintf(ArqSai, "-");
	  }
	}
	fprintf(ArqSai, "\n");
	for (j=0; j<SeqDesm-1; j++){
	  fprintf(ArqSai, "     ");
	}
	fprintf(ArqSai, "    \\ ");
	for (k=0; k<SeqMont-SeqDesm-2; k++){
	  fprintf(ArqSai, "     ");
	}
	fprintf(ArqSai, "   / \n");
	for (j=0; j<SeqDesm; j++){
	  fprintf(ArqSai, "     ");
	}
	for (k=0; k<SeqMont-SeqDesm-1; k++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	  if (k < SeqMont-SeqDesm-2){
	    fprintf(ArqSai, "-");
	  }
	  a = a+1;
	}
      }
      else{
	fprintf(ArqSai, "-");
	for (k=0; k<SeqMont-SeqDesm-1; k++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	  fprintf(ArqSai, "-");
	  a = a+1;
	}
	for (j=(SeqDesm+(RamifJob*(SeqMont-SeqDesm-1))); j<NumOPJ; j++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][j]-1);
	  if (j < NumOPJ-1){
	    fprintf(ArqSai, "-");
	  }
	}
	fprintf(ArqSai, "\n");
	for (j=0; j<SeqDesm-1; j++){
	  fprintf(ArqSai, "     ");
	}
	fprintf(ArqSai, "    \\ ");
	for (k=0; k<SeqMont-SeqDesm-2; k++){
	  fprintf(ArqSai, "     ");
	}
	fprintf(ArqSai, "   / \n");
	for (j=0; j<SeqDesm; j++){
	  fprintf(ArqSai, "     ");
	}
	for (k=0; k<SeqMont-SeqDesm-1; k++){
	  fprintf(ArqSai, "%3d ", MatOpeJob[i][a]-1);
	  if (k < SeqMont-SeqDesm-2){
	    fprintf(ArqSai, "-");
	  }
	  a = a+1;
	}
      }
    }
   }
  
  fclose(ArqSai);
  for(i=0; i<NumJob; i++){
    free(MatOpeJob[i]);
  }
  free(MatOpeJob);
  for(i=0; i<NumJob*NumMaq*3; i++){
    free(MatMaqAlt[i]);
  }
  free(MatMaqAlt);
  for(i=0; i<NumJob*NumMaq*3; i++){
    free(MatTemPro[i]);
  }
  free(MatTemPro);
}

int main(){
  int i;
  
  printf("Number of combinations (jobs X machines) to create: ");
  scanf("%d", &NumGru);
  for (i=0; i<NumGru; i++){
    printf("Number of jobs of combination %d: ", i+1);
    scanf("%d", &LisPar[i][0]);
    printf("Number of machines of combination %d (5 or more): ", i+1);
    scanf("%d", &LisPar[i][1]);
    while (LisPar[i][1] < 5){
      printf("(Number of machines must be 5 or more to ensure correct DA jobs.)\n");
      printf("Number of machines of combination %d: ", i+1);
      scanf("%d", &LisPar[i][1]);
    }
    printf("Number of instances of combination %d: ", i+1);
    scanf("%d", &LisPar[i][2]);
  }
  printf("Instance file name (max. 100 chars): ");
  scanf("%s", &NomSai2);

  Flex = 0.5;
  NumIns = 0;
  for (Cont=0; Cont<NumGru; Cont++){
    for (i=0; i<LisPar[Cont][2]; i++){
      GerIns();
      NumIns = NumIns+1;
    }
  }
  return 0;
}

