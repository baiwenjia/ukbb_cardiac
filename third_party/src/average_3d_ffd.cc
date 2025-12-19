#include <mirtk/Common.h>
#include <mirtk/Options.h>
#include <mirtk/IOConfig.h>
#include <mirtk/GenericImage.h>
#include <mirtk/Transformations.h>

using namespace mirtk;

// =============================================================================
// Help
// =============================================================================

// -----------------------------------------------------------------------------
// Print help screen
void PrintHelp(const char *name)
{
  cout << endl;
  cout << "Usage: " << name << " n <T1> <w1> <T2> <w2>... <Tn> <wn> <T_out>" << endl;
  cout << endl;
  
  cout << "Description:" << endl;
  cout << "  Computes the average T of the given input free-form deformations such that" << endl;
  cout << "  T_out(x) = T1(x) * w1 + T2(x) * w2 ... + Tn(x) * wn " << endl;
  cout << endl;

  cout << "<options> can be one or more of the following:" << endl;
  cerr << "<-verbose>    Display information." << endl;
  exit(1);
}

// =============================================================================
// Main
// =============================================================================

// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  // Parse arguments
  char *command = argv[0];
  argc--; argv++;
  if (argc < 3) {
    PrintHelp(command);
  }

  int n_trans = atoi(argv[0]);
  argc--; argv++;
  char **input_name = new char *[n_trans];
  double *w = new double[n_trans];
  for (int i = 0; i < n_trans; i++) {
    input_name[i] = argv[0];
    argc--; argv++;
    w[i] = atof(argv[0]);
    argc--; argv++;
  }
  
  char *output_name = argv[0];
  argc--; argv++;

  bool verbose = false;
  while(argc > 0){
    bool ok = false;
    if((ok == false) && (strcmp(argv[0], "-verbose") == 0)){
      argc--; argv++;
      verbose = true;
      ok = true;
    }
    if(ok == false){
      cerr << "Can not parse argument " << argv[0] << endl;
      PrintHelp(command);
    }
  }

  // Read the input transformations
  MultiLevelTransformation **T = new MultiLevelTransformation *[n_trans];
  for (int i = 0; i < n_trans; i++) {
    if (verbose) {
      cout << "Reading transformation " << i << " from " << input_name[i] << " (weight = " << w[i] << ") ..." << endl;
    }
    Transformation *ptr = Transformation::New(input_name[i]);
    T[i] = dynamic_cast<MultiLevelTransformation *>(ptr);
    if (T[i] == NULL) {
      cout << "Error: error in reading the transformation file." << endl;
      exit(0);
    }
    if (T[i]->NumberOfLevels() > 1) {
      cout << "Error: the transformation has more than one local FFDs." << endl;
      exit(0);
    }
  }

  // Get information from the first ffd
  BSplineFreeFormTransformation3D *T0_ffd = dynamic_cast<BSplineFreeFormTransformation3D *>(T[0]->GetLocalTransformation(0));
  ImageAttributes attr = T0_ffd->Attributes();
  int X = attr._x;
  int Y = attr._y;
  int Z = attr._z;

  // Allocate the output ffd
  BSplineFreeFormTransformation3D *out_ffd = new BSplineFreeFormTransformation3D(attr);

  // For each control point
  for (int k = 0; k < Z; k++) {
    for (int j = 0; j < Y; j++) {
      for (int i = 0; i < X; i++) {
        double out_dx = 0;
        double out_dy = 0;
        double out_dz = 0;
          
        for (int n = 0; n < n_trans; n++) {
          double dx, dy, dz;
          T[n]->GetLocalTransformation(0)->Get(i, j, k, dx, dy, dz);
          out_dx += w[n] * dx;
          out_dy += w[n] * dy;
          out_dz += w[n] * dz;
        }
        
        out_ffd->Put(i, j, k, out_dx, out_dy, out_dz);
      }
    }
  }

  // The output transformation
  MultiLevelFreeFormTransformation *T_out = new MultiLevelFreeFormTransformation;
  T_out->PushLocalTransformation(out_ffd);
  T_out->Write(output_name);
  if (verbose) {
    cout << "Writing the average transformation to " << output_name << endl;
  }
}
