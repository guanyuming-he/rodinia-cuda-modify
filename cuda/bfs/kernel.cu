/*********************************************************************************
Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void
Kernel( Node* g_graph_nodes, int* g_graph_edges_startend, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int edge_list_size) 
{
	int tid = blockIdx.x * 1024 + threadIdx.x;
	if( tid<edge_list_size)
	{
		int start = g_graph_edges_startend[2*tid];
		int end = g_graph_edges_startend[2*tid+1];

		if (g_graph_mask[start]) // is start in q2?
		{
			g_graph_mask[start]=false;
			if(!g_graph_visited[end])
			{
				g_cost[end]=g_cost[start]+1;
				g_updating_graph_mask[end]=true;
			}
		}
	}
}

#endif 
