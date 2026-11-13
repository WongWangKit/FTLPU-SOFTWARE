#include "ftlpu/compiler/Analysis/planar_access.hpp"
#include <cstdlib>
#include <iostream>

using namespace ftlpu::compiler::analysis;
static void check(bool ok,const char *message) {
  if(!ok){std::cerr<<message<<'\n';std::exit(1);}
}
int main() {
  // Cover whole matrix views, sliced heads, repeated tokens, reversed rows,
  // column slices and nested views. Expected addresses come from the source
  // layout equations, independently of the READ3D factorization chosen.
  for(int rows:{2,4,16,32,64})for(int columns:{1,2,4})
    for(int chunk=1;chunk<=rows;chunk*=2)for(int inner:{0,1,-1,4}) {
      std::vector<int64_t> addresses;
      for(int c=0;c<columns;++c)for(int r=0;r<rows;++r)
        addresses.push_back(512+c*192+(r/chunk)*77+(r%chunk)*inner);
      auto plan=planPlanarRead(rows,columns,addresses);
      check(bool(plan),"valid affine view rejected");
      for(int c=0;c<columns;++c)for(int r=0;r<rows;++r)
        check(plan->address(r,c)==addresses[c*rows+r],"view changed physical word");
      if(rows>=4) {
        auto part=slicePlanarRows(*plan,rows/2,rows/2);
        check(bool(part),"aligned nested row view rejected");
        for(int c=0;c<columns;++c)for(int r=0;r<rows/2;++r)
          check(part->address(r,c)==addresses[c*rows+r+rows/2],"nested view changed word");
      }
    }
  check(!planPlanarRead(3,1,{0,1,7}),"nonaffine view accepted");
  check(!planPlanarRead(2,2,{0,1,8,11}),"nonseparable columns accepted");
  check(!planPlanarRead(2,1,{-1,0}),"negative physical row accepted");
  check(!planPlanarRead(0,1,{}),"empty domain accepted");
  auto one=planPlanarRead(2,1,{0,1});
  check(!slicePlanarRows(*one,1,2),"out-of-bounds slice accepted");
  std::cout<<"planar access contracts passed\n";
}
