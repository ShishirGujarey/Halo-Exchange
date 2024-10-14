#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "mpi.h"
#include <string.h>
// mpirun -np P -f hostfile ./halo Px N <num_time_steps> <seed> <stencil>
// #pragma prutor-mpi-args: -np 12 -ppn 6

//  TC-1 : #pragma prutor-mpi-sysargs: 4 262144 10 7 5 
//  TC-2 : #pragma prutor-mpi-sysargs: 4 262144 10 7 9
//  TC-3 : #pragma prutor-mpi-sysargs: 4 4194304 10 7 5
//  TC-4 : #pragma prutor-mpi-sysargs: 4 4194304 10 7 9

int main( int argc, char *argv[])
{
    MPI_Init (&argc, &argv);
    int myrank,P;
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
   
    MPI_Status status;
    MPI_Request request;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&P);
    
    int Px = atoi(argv[1]);
    int N = atoi(argv[2]);
    int n = sqrt((double)N);
    int steps = atoi(argv[3]);
    int stencil = atoi(argv[5]);
    
    double buf[n][n];
    int seed = atoi(argv[4]);
    srand(seed*(myrank+10));

    // Initialize data
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
        {
            buf[i][j] = abs(rand()+(i*rand()+j*myrank))/100;
        }
    }
    int Py = P/Px;
    
    //Position of current process
    int p_i = myrank/Py;
    int p_j = myrank%Py;

    //Determine where communication is to be done
    int left = 1, right = 1, up = 1, bot = 1;
    if (p_j == 0)       left = 0;
    if (p_j == Py-1)    right = 0;
    if (p_i == 0)       up = 0;
    if (p_i == Px-1)    bot = 0;

    //Store boundary points, buffers for them
    double arr_left[n] ,arr_right[n],arr_up[n] ,arr_bot[n] ;
    double arr_left_2[2*n] ,arr_right_2[2*n] ,arr_up_2[2*n] ,arr_bot_2[2*n];
    memset( arr_left, 0,   n*sizeof(double) );
    memset( arr_right, 0,   n*sizeof(double) );
    memset( arr_up, 0,   n*sizeof(double) );
    memset( arr_bot, 0,   n*sizeof(double) );
    memset( arr_left_2, 0, 2*n*sizeof(double) );
    memset( arr_right_2, 0, 2*n*sizeof(double) );
    memset( arr_up_2, 0, 2*n*sizeof(double) );
    memset( arr_bot_2, 0, 2*n*sizeof(double) );
    double send_left[n],send_right[n],send_up[n],send_bot[n];
    double send_left_2[2*n],send_right_2[2*n],send_up_2[2*n],send_bot_2[2*n];
    double recv_left[n],recv_right[n],recv_up[n],recv_bot[n];
    double recv_left_2[2*n],recv_right_2[2*n],recv_up_2[2*n],recv_bot_2[2*n];

    int t = 0;
    MPI_Barrier( MPI_COMM_WORLD);
    double sTime = MPI_Wtime();

    while(t<steps){
        //Step 1: Communication

        //Pack corresponding row/column and send
        int pos;
        if(up){
            pos = 0;
            if(stencil == 5){
                for(int i=0; i<n; i++) MPI_Pack(&buf[0][i], 1, MPI_DOUBLE, send_up, n*sizeof(double), &pos, MPI_COMM_WORLD); 
                
                MPI_Isend(send_up, pos, MPI_PACKED, myrank-Py, myrank, MPI_COMM_WORLD, &request);
            }

            if(stencil == 9){
                for(int i=0; i<n; i++) MPI_Pack(&buf[0][i], 1, MPI_DOUBLE, send_up_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
                for(int i=0; i<n; i++) MPI_Pack(&buf[1][i], 1, MPI_DOUBLE, send_up_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
                
                MPI_Isend(send_up_2, pos, MPI_PACKED, myrank-Py, myrank, MPI_COMM_WORLD, &request);
            }
        }
        if(bot){
            pos=0;
            if(stencil == 5){
                for(int i=0; i<n; i++) MPI_Pack(&buf[n-1][i], 1, MPI_DOUBLE, send_bot, n*sizeof(double), &pos, MPI_COMM_WORLD);
                    
                MPI_Isend(send_bot, pos, MPI_PACKED, myrank+Py, myrank, MPI_COMM_WORLD, &request);           
                }
            
            if (stencil == 9){
                for(int i=0; i<n; i++) MPI_Pack(&buf[n-1][i], 1, MPI_DOUBLE, send_bot_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
                for(int i=0; i<n; i++) MPI_Pack(&buf[n-2][i], 1, MPI_DOUBLE, send_bot_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
                
                MPI_Isend(send_bot_2, pos, MPI_PACKED, myrank+Py, myrank, MPI_COMM_WORLD, &request); 
            }
        }
        if(left){
            pos=0;
            if(stencil == 5){
                for(int i=0; i<n; i++) MPI_Pack(&buf[i][0],1,MPI_DOUBLE,send_left,n*sizeof(double),&pos,MPI_COMM_WORLD);

                MPI_Isend(send_left,pos,MPI_PACKED,myrank-1,myrank,MPI_COMM_WORLD,&request);
            }
            if (stencil == 9){
                for(int i=0; i<n; i++) MPI_Pack(&buf[i][0],1,MPI_DOUBLE,send_left_2, 2*n*sizeof(double),&pos,MPI_COMM_WORLD);
                for(int i=0; i<n; i++) MPI_Pack(&buf[i][1],1,MPI_DOUBLE,send_left_2, 2*n*sizeof(double),&pos,MPI_COMM_WORLD);

                MPI_Isend(send_left_2,pos,MPI_PACKED,myrank-1,myrank,MPI_COMM_WORLD,&request);
            }
        }
        if(right){
            pos=0;
            if(stencil == 5){
                for(int i=0; i<n; i++) MPI_Pack(&buf[i][n-1], 1, MPI_DOUBLE, send_right, n*sizeof(double), &pos, MPI_COMM_WORLD);
                
                MPI_Isend(send_right, pos, MPI_PACKED, myrank+1, myrank, MPI_COMM_WORLD, &request);
            }
            if (stencil == 9){
                for(int i=0; i<n; i++) MPI_Pack(&buf[i][n-1], 1, MPI_DOUBLE, send_right_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
                for(int i=0; i<n; i++) MPI_Pack(&buf[i][n-2], 1, MPI_DOUBLE, send_right_2, 2*n*sizeof(double), &pos, MPI_COMM_WORLD);
                
                MPI_Isend(send_right_2, pos, MPI_PACKED, myrank+1, myrank, MPI_COMM_WORLD, &request);
            }
        }

        //Receiving data from neighbouring processes (Unpack and store in corresponding arrays)
        if(up){
            pos=0;
            if (stencil == 5){
                MPI_Recv(recv_up, n*sizeof(double), MPI_PACKED, myrank-Py, myrank-Py, MPI_COMM_WORLD, &status);
                for(int i=0; i<n; i++) MPI_Unpack(recv_up, n*sizeof(double), &pos, arr_up+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
            if (stencil == 9){
                MPI_Recv(recv_up_2, 2*n*sizeof(double), MPI_PACKED, myrank-Py, myrank-Py, MPI_COMM_WORLD, &status);
                for(int i=0; i<2*n; i++) MPI_Unpack(recv_up_2, 2*n*sizeof(double), &pos, arr_up_2+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }
        if(bot){
            pos=0;
            if (stencil == 5){
                MPI_Recv(recv_bot,n*sizeof(double),MPI_PACKED,myrank+Py,myrank+Py,MPI_COMM_WORLD,&status);
                for(int i=0; i<n; i++) MPI_Unpack(recv_bot, n*sizeof(double), &pos, arr_bot+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
            if (stencil == 9){
                MPI_Recv(recv_bot_2,2*n*sizeof(double),MPI_PACKED,myrank+Py,myrank+Py,MPI_COMM_WORLD,&status);
                for(int i=0; i<2*n; i++) MPI_Unpack(recv_bot_2, 2*n*sizeof(double), &pos, arr_bot_2+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }
        if(left){
            pos=0;
            if (stencil == 5){
                MPI_Recv(recv_left, n*sizeof(double), MPI_PACKED, myrank-1, myrank-1, MPI_COMM_WORLD, &status);
                for(int i=0; i<n; i++) MPI_Unpack(recv_left, n*sizeof(double), &pos, arr_left+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
            if (stencil == 9){
                MPI_Recv(recv_left_2, 2*n*sizeof(double), MPI_PACKED, myrank-1, myrank-1, MPI_COMM_WORLD, &status);
                for(int i=0; i<2*n; i++) MPI_Unpack(recv_left_2, 2*n*sizeof(double), &pos, arr_left_2+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }
        if(right){
            pos=0;
            if (stencil == 5){
                MPI_Recv(recv_right, n*sizeof(double), MPI_PACKED, myrank+1, myrank+1, MPI_COMM_WORLD, &status);
                for(int i=0; i<n; i++) MPI_Unpack(recv_right, n*sizeof(double), &pos, arr_right+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
            if (stencil == 9){
                MPI_Recv(recv_right_2, 2*n*sizeof(double), MPI_PACKED, myrank+1, myrank+1, MPI_COMM_WORLD, &status);
                for(int i=0; i<2*n; i++) MPI_Unpack(recv_right_2, 2*n*sizeof(double), &pos, arr_right_2+i, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }
        
        //Step 2: Averaging
        double temp[n][n];
        if (stencil == 5){
            int d = 5;

            //Corner points
            if (!left || !up) d=4; 
            if (!left && !up) d=3;
            temp[0][0] = (arr_up[0] + arr_left[0] + buf[0][1] + buf[1][0] + buf[0][0])/d;
            d=5;

            if (!right || !up) d= 4;  
            if (!right && !up) d= 3;  
            temp[0][n-1]   = (arr_up[n-1] + arr_right[0] + buf[0][n-2] + buf[1][n-1] + buf[0][n-1])/d;
            d = 5;

            if (!left || !bot) d= 4;
            if (!left && !bot) d= 3;
            temp[n-1][0]   = (arr_bot[0] + arr_left[n-1] + buf[n-1][1] + buf[n-2][0] + buf[n-1][0])/d;
            d = 5;

            if (!bot || !right) d = 4; 
            if (!bot && !right) d = 3; 
            temp[n-1][n-1] = (arr_bot[n-1] + arr_right[n-1] + buf[n-2][n-1] + buf[n-1][n-2] + buf[n-1][n-1])/d;        
            d=5;

            //Edges
            for(int i=1; i<n-1; i++)
            {
                if(!up) d = 4;
                temp[0][i]   = (buf[0][i+1] + buf[0][i-1] + buf[1][i] + arr_up[i] + buf[0][i])/d;
                d= 5;
                if(!bot) d = 4;
                temp[n-1][i] = (buf[n-1][i+1] + buf[n-1][i-1] + buf[n-2][i] + arr_bot[i] + buf[n-1][i])/d;
                d=5;
                if(!left) d = 4;
                temp[i][0]   = (buf[i+1][0] + buf[i-1][0] + buf[i][1] + arr_left[i] + buf[i][0])/d;
                d = 5;
                if(!right) d = 4;
                temp[i][n-1] = (buf[i+1][n-1] + buf[i-1][n-1] + buf[i][n-2] + arr_right[i] + buf[i][n-1])/d;
                d=5;
            }
            //Interior points
            for(int i=1;i<n-1;i++){
                for(int j=1;j<n-1;j++)
                {
                    temp [i][j] = (buf[i-1][j] + buf[i+1][j] + buf[i][j-1] + buf[i][j+1] + buf[i][j])/5;
                }
            }
        }

        if (stencil == 9){
            //Corner points
            int d = 9;
            if (!left || !up) d=7; 
            if (!left && !up) d=5;
            temp[0][0]     = (arr_up_2[0] + arr_up_2[n] + arr_left_2[0] + arr_left_2[n] + buf[0][1] + buf[0][2] + buf[1][0]+ buf[2][0] + buf[0][0])/d;
            d = 9;
            if (!right || !up) d= 7;  
            if (!right && !up) d= 5; 
            temp[0][n-1]   = (arr_up_2[n-1] + arr_up_2[2*n-1] + arr_right_2[0] + arr_right_2[n] + buf[0][n-2] + buf[0][n-3] + buf[1][n-1] + buf[2][n-1] + buf[0][n-1])/d;
            d = 9;
            if (!left || !bot) d= 7;
            if (!left && !bot) d= 5;
            temp[n-1][0]   = (arr_bot_2[0] + arr_bot_2[n] + arr_left_2[n-1]+arr_left_2[2*n-1] + buf[n-1][1] + buf[n-2][0] + buf[n-3][0] + buf[n-1][2] + buf[n-1][0])/d;
            d = 9;
            if (!bot || !right) d = 7; 
            if (!bot && !right) d = 5; 
            temp[n-1][n-1] = (arr_bot_2[n-1] + arr_bot_2[2*n-1] + arr_right_2[n-1] + arr_right_2[2*n-1] + buf[n-2][n-1] + buf[n-3][n-1] + buf[n-1][n-2] + buf[n-1][n-3] + buf[n-1][n-1])/d;  
            d = 9;

            //Second to Corner
            if (!left || !up) d=8; 
            if (!left && !up) d=7;
            temp[1][1]     = (buf[0][1] + buf[2][1] + buf[3][1] + buf[1][0] + buf[1][2] + buf[1][3] + arr_up_2[1] + arr_left_2[1] + buf[1][1])/d;
            d = 9;
            if (!right || !up) d= 8;  
            if (!right && !up) d= 7; 
            temp[1][n-2]   = (buf[0][n-2] + buf[2][n-2] + buf[3][n-2] + buf[1][n-4] + buf[1][n-3] + buf[1][n-1] + arr_up_2[n-2] + arr_right_2[1] + buf[1][n-2])/d;
            d = 9;
            if (!left || !bot) d= 8;
            if (!left && !bot) d= 7;
            temp[n-2][1]   = (buf[n-4][1] + buf[n-3][1] + buf[n-1][1] + buf[n-2][0] + buf[n-2][2] + buf[n-2][3] + arr_bot_2[1] + arr_left_2[n-2] + buf[n-2][1])/d;
            d = 9;
            if (!bot || !right) d = 8; 
            if (!bot && !right) d = 7; 
            temp[n-2][n-2] = (buf[n-2][n-1] + buf[n-2][n-3] + buf[n-2][n-4] + buf[n-1][n-2] + buf[n-3][n-2] + buf[n-4][n-2] + arr_bot_2[n-2] + arr_right_2[n-2] + buf[n-2][n-2])/d;
            d = 9;

            if (!up) d = 7;
            if (!left) d = 8;
            if (!left && !up) d = 6;
            temp[0][1]     = (buf[0][2] + buf[0][3] + buf[0][0] + buf[1][1] + buf[2][1] + arr_left_2[0] + arr_up_2[1] + arr_up_2[1+n] + buf[0][1])/d;
            d = 9;
            if (!up) d = 8;
            if (!left) d = 7;
            if (!left && !up) d = 6;
            temp[1][0]     = (buf[0][0] + buf[2][0] + buf[3][0] + buf[1][2] + buf[1][1]+arr_up_2[0] + arr_left_2[1] + arr_left_2[1+n] + buf[1][0])/d;
            d = 9;
            if (!up) d = 7;
            if (!right) d = 8;
            if (!right && !up) d = 6;
            temp[0][n-2]   = (buf[1][n-2] + buf[2][n-2] + buf[0][n-4] + buf[0][n-3] + buf[0][n-1] + arr_up_2[2*n-2] + arr_up_2[n-2] + arr_right_2[0] + buf[0][n-2])/d;
            d = 9;
            if (!up) d = 8;
            if (!right) d = 7;
            if (!right && !up) d = 6;
            
            temp[1][n-1]   = (buf[0][n-1] + buf[2][n-1] + buf[3][n-1] + buf[1][n-2] + buf[1][n-3] + arr_up_2[n-1] + arr_right_2[1+n] + arr_right_2[1] + buf[1][n-1])/d;
            d = 9;
            if (!bot) d = 8;
            if (!left) d = 7;
            if (!left && !bot) d = 6;
            temp[n-2][0]   = (buf[n-4][0] + buf[n-3][0] + buf[n-1][0] + buf[n-2][1] + buf[n-2][2] + arr_bot_2[0] + arr_left_2[n-2] + arr_left_2[2*n-2] + buf[n-2][0])/d;
            d = 9;
            if (!bot) d = 7;
            if (!left) d = 8;
            if (!left && !bot) d = 6;
            temp[n-1][1]   = (buf[n-2][1] + buf[n-3][1] + buf[n-1][0] + buf[n-1][3] + buf[n-1][2] + arr_bot_2[1] + arr_bot_2[n+1] + arr_left_2[n-1] + buf[n-1][1])/d;
            d = 9;
            if (!bot) d = 8;
            if (!right) d = 7;
            if (!right && !bot) d = 6;
            temp[n-2][n-1] = (buf[n-2][n-3] + buf[n-2][n-2] + buf[n-1][n-1] + buf[n-3][n-1] + buf[n-4][n-1] + arr_bot_2[n-1] + arr_right_2[n-2] + arr_right_2[2*n-2] + buf[n-2][n-1])/d;
            d = 9;
            if (!bot) d = 7;
            if (!right) d = 8;
            if (!right && !bot) d = 6;
            temp[n-1][n-2] = (buf[n-1][n-3] + buf[n-1][n-4] + buf[n-1][n-1] + buf[n-3][n-2] + buf[n-2][n-2] + arr_bot_2[n-2] + arr_bot_2[2*n-2] + arr_right_2[n-1] + buf[n-1][n-2])/d;
            d = 9;

            //Edges
            for(int i=2; i<n-2; i++)
            {
                d = 9;
                if(!up) d = 7;
                temp[0][i]   = (buf[0][i+1] + buf[0][i+2] + buf[0][i-1] + buf[0][i-2] + buf[1][i] + buf[2][i] + arr_up_2[i] + arr_up_2[i+n] + buf[0][i])/d;
                d = 9;
                if(!up) d = 8;
                temp[1][i]   = (buf[1][i+1] + buf[1][i+2] + buf[1][i-1] + buf[1][i-2] + buf[0][i] + buf[2][i] + buf[3][i] + arr_up_2[i] + buf[1][i])/d;
                d = 9;
                if(!bot) d = 7;
                temp[n-1][i] = (buf[n-1][i+1] + buf[n-1][i+2] + buf[n-1][i-1]+buf[n-1][i-2] + buf[n-3][i]+ buf[n-2][i] + arr_bot_2[i]+arr_bot_2[i+n]+ buf[n-1][i])/d;
                d = 9;
                if(!bot) d = 8;
                temp[n-2][i] = (buf[n-2][i+1] + buf[n-2][i+2] + buf[n-2][i-1]+buf[n-2][i-2] + buf[n-1][i] + buf[n-3][i] + buf[n-4][i] + arr_bot_2[i] + buf[n-2][i])/d;
                d = 9;
                if(!left) d = 7;
                temp[i][0]   = (buf[i+1][0]+buf[i+2][0]+buf[i-2][0] + buf[i-1][0] + buf[i][1]+buf[i][2] + arr_left_2[i]+ arr_left_2[i+n] + buf[i][0])/d;
                d = 9;
                if(!left) d = 8;
                temp[i][1]   = (buf[i+1][1]+buf[i+2][1]+buf[i-2][1] + buf[i-1][1] + buf[i][0]+buf[i][2] + buf[i][3] + arr_left_2[i] + buf[i][1])/d;
                d = 9;
                if(!right) d = 7;
                temp[i][n-1] = (buf[i+1][n-1]+buf[i+2][n-1] + buf[i-2][n-1]+ buf[i-1][n-1] + buf[i][n-2] +buf[i][n-3]+ arr_right_2[i] + arr_right_2[i+n]+ buf[i][n-1])/d;
                d = 9;
                if(!right) d = 8;
                temp[i][n-2] = (buf[i+1][n-2]+buf[i+2][n-2] + buf[i-2][n-2]+ buf[i-1][n-2] + buf[i][n-3] +  buf[i][n-1]+buf[i][n-4]+ + arr_right_2[i] + buf[i][n-2])/d;
                d = 9;
            }

            //Interior points
            for(int i=2;i<n-2;i++){
                for(int j=2;j<n-2;j++)
                {
                    temp[i][j] = (buf[i-1][j] + buf[i-2][j] + buf[i+1][j] + buf[i+2][j] + buf[i][j-1] + buf[i][j-2] + buf[i][j+1] + buf[i][j+2] + buf[i][j])/9;
                }
            }
        }
        //Incrementing the steps, copying temporary data to original matrix
        t++;
        for(int i = 0; i<n; i++){
            for (int j= 0; j<n; j++){
                buf[i][j] = temp[i][j];
            }
        }
        
    }
    double eTime = MPI_Wtime();
    double time = eTime-sTime;
    double maxTime;

    //Get maximum of time taken by all processes
    MPI_Reduce (&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if(myrank == 0){
        printf("Time taken: %lf\n",maxTime);
    }

    MPI_Finalize();
    return 0;
}