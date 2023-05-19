#include <stdio.h>

extern ilaver_(int* major,int* minor,int* patch);

/**
 * Determines the LAPACK library version used in C.
 * Use for debugging and solving problems with ITensor installs and other LAPACK dependents.
*/
int main()
{
    int major=0;
    int minor=0;
    int patch=0;

    ilaver_(&major,&minor,&patch);

    printf("lapack %d.%d.%d\n",major,minor,patch);
}
