
import sys;
sys.path.insert(0,"../");
sys.path.insert(0,"../core/");
import pycuda.driver as cuda;
import pycuda.autoinit;
import pycuda.compiler;
from pycuda.curandom import rand as curand;
from pycuda.elementwise import ElementwiseKernel;
import pycuda.gpuarray as gpuarray;
from pycuda import cumath; 


import numpy;


import constants;
import netParams;

"""
Cuda interface for computing on gpu

"""

debug = False;

#SOURCE CODE FOR PYTHON
mod = pycuda.compiler.SourceModule(
"""
    //put in range
    __global__ void putinrange(float *SOL)
    {
        int idx = threadIdx.x;
        
        if(SOL[idx] < -1.0)
            SOL[idx] = 0.0;
        
        if(SOL[idx] > 1.0)
            SOL[idx] = 1.0;
   
    }
    
    //cross-over operation
    __global__ void crossOver(float *SOLA, float *SOLB)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        
        //cross-over genes between solutions

        float g = SOLA[idx];
        float g2 = SOLB[idx];
        
        SOLB[idx] = g;
        SOLA[idx] = g2;
        
    }
    
    //differential evolution - float *FOLLOWER_SOLS, float *TO_MUTATE_SOLS,
    __global__ void differential_evolve(float *LEADER_SOLS)
    {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        
        
        LEADER_SOLS[idx] = LEADER_SOLS[idx] * 2;
        
        
        LEADER_SOLS[idy] = LEADER_SOLS[idy] * 2;
        
        
    }
    
"""
);




def cuda_putInRange(sols,minVal, maxVal):
    """ mutatest the given genetic string using cuda"""
    

    a = numpy.random.randn(4).astype(numpy.float32);
    
    #allocate memory to device
    a_gpu = cuda.mem_alloc(a.nbytes);
    minVal_gpu = cuda.mem_alloc(sys.getsizeof(minVal));
    minVal_gpu = cuda.mem_alloc(sys.getsizeof(maxVal));
    
    #copy to memory
    cuda.memcpy_htod(a_gpu, a);
    #cuda.memcpy_htod(minVal_gpu, minVal);
    #cuda.memcpy_htod(maxVal_gpu, maxVal);
    
    
    func = mod.get_function("putinrange");
    func(a_gpu, block=(4,4,1));
    
    a_doubled = numpy.empty_like(a);
    cuda.memcpy_dtoh(a_doubled, a_gpu);

def cuda_ageSols(sols):
    """ makes solutions to age """

    #get num sols
    num_sols = len(sols);
    
    
    
    #convert to form of numpy arrays
    sols_arr = numpy.array(sols, numpy.float32);
    ones_arr = numpy.zeros_like(sols,numpy.float32);
    ones_arr[:,constants.AGE_GENE] = 1;
    
    #copy each to gpu
    sols_gpu = gpuarray.to_gpu(sols_arr);
    mask_gpu = gpuarray.to_gpu(ones_arr);
    
    #debug
    if debug == True:
        print mask_gpu.view();
    
    #apply mask
    aged_sols_gpu = sols_gpu + mask_gpu;
    
    sols = aged_sols_gpu.get().tolist();
    
    
    
def cuda_crossOver(sola, solb):
    """ """
    
    sol_len = len(sola);
    
    a_gpu = cuda.mem_alloc(sola.nbytes);
    b_gpu = cuda.mem_alloc(solb.nbytes);
    
    cuda.memcpy_htod(a_gpu, sola);
    cuda.memcpy_htod(b_gpu, solb);
    
    func = mod.get_function("crossOver");
    func(a_gpu,b_gpu, block=(sol_len,1,1));
    
    a_new = numpy.empty_like(sola);
    b_new = numpy.empty_like(solb);
    
    cuda.memcpy_dtoh(a_new, a_gpu);
    cuda.memcpy_dtoh(b_new, b_gpu);
    
    if debug == True:
        print "a:", a;
        print "b:",b;
        print "new a:",a_new;
        print "new b:",b_new;
        
    return a_new,b_new;

def cuda_diffEvo(leader_sols, follower_sols, to_mutate_sols, alpha):
    """ performs differential evolution """
    
    print ">>GPU DIFF EVOLVE in progress...";
    
    #LEADER_SOLS , FOLLOWER_SOLS AND TO_MUTATE ALL HAVE TO BE OF THESAME LENGTH
    sol_len = len(leader_sols[0]);
 
    
    #get number of nodes
    num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
    
    #operation point
    diffEvolveFrom = constants.META_INFO_COUNT + num_nodes;
    
    #convert to form of numpy arrays
    mutants = numpy.array(to_mutate_sols, numpy.float32);
    #-copy only the parts to be mutated
    leaders = numpy.array(leader_sols[:,diffEvolveFrom:], numpy.float32);
    followers = numpy.array(follower_sols[:,diffEvolveFrom:], numpy.float32);
    to_mutate = numpy.array(to_mutate_sols[:,diffEvolveFrom:], numpy.float32);
    
    #store velocities
    velocities = numpy.zeros_like(to_mutate).astype(numpy.float32);
    
    #copy each to gpu
    L_gpu = gpuarray.to_gpu(leaders);
    F_gpu = gpuarray.to_gpu(followers);
    M_gpu = gpuarray.to_gpu(to_mutate);
    Vel_gpu = gpuarray.zeros_like(M_gpu);
    GVel_gpu = gpuarray.zeros((sol_len),numpy.float32);
    
    
    #calculate velocities
    Vel_gpu = alpha * (L_gpu - F_gpu);
    
    #caluate grand velocities
    GVel_gpu = Vel_gpu * 1/len(Vel_gpu);
    
    #add to mutant solutions
    M_gpu = M_gpu + Vel_gpu;
    
    #retrieve from gpu
    mutations = M_gpu.get();
    #copy back mutations
    mutants[:,diffEvolveFrom:] = mutations;
    
    gvel = GVel_gpu.get();
    vel = Vel_gpu.get();
    
    if debug == True:
        print "\n* before operation";
        print "leader_sols:", leader_sols;
        print "follower_solsL", follower_sols;
        print "to_mutate_sols", to_mutate_sols;
        
        print "\n* after operation";
        print "velocities(gpu output)", vel;
        print "grandVel(gpu output):", gvel;
        print "mutants(gpu output):", mutants;
    
    
    return mutants.tolist();

def cuda_mutate(sols,prob_mut, mut_range,min_param,max_param):
    """ mutates the values of the solutions given
    @params sols, probability of mutation, mutation range, min param, max param
    @returns mutated sols
    """

    #ALL SOLUTIONS MUST BE OF SAME LENGTH
    num_sols = len(sols);
    #get length of solutions
    sol_len = len(sols[0]);
    

    
    #get number of nodes
    num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
    
    #mutate not on architecture
    mutateFrom = constants.META_INFO_COUNT + num_nodes;
        
    
    #range
    m_range = 2 * mut_range;
    
    #convert to form of numpy arrays
    old_sols = numpy.array(sols[:,mutateFrom:], numpy.float32);
    cost_genes = numpy.ones((num_sols),numpy.float32);
    contrb_genes = numpy.zeros((num_sols),numpy.float32);
    mutants = numpy.array(sols).astype(numpy.float32);
    cost_genes *= -1;
    age_genes = numpy.zeros((num_sols),numpy.float32);
    
    
    
    #copy to gpu
    sols_gpu = gpuarray.to_gpu(old_sols);
    sol_len = len(old_sols[0]);
    
    #operation
    MutSols_gpu = gpuarray.zeros_like(sols_gpu).astype(numpy.float32);
    Mvals_gpu = (curand((num_sols,sol_len),numpy.float32) * m_range) - mut_range; #mutation values
    
    #calculate probabilites of mutation and form mutation mask
    Mprob_gpu = curand((num_sols,sol_len),numpy.float32); #mutation probabilities
    MutMask_gpu = gpuarray.zeros_like(Mprob_gpu).astype(numpy.float32);
    #-form mutation    
    form_mutation_mask(Mprob_gpu,MutMask_gpu,prob_mut);
    #-mutate genes
    MutSols_gpu = sols_gpu + (MutMask_gpu * Mvals_gpu);
    
    #get mutated solutions
    mutants[:,mutateFrom:] = MutSols_gpu.get();
    mutants[:,constants.COST_GENE] = cost_genes;
    mutants[:,constants.COST2_GENE] = cost_genes;
    mutants[:,constants.MISC_GENE] = contrb_genes;
    mutants[:,constants.AGE_GENE] = age_genes;
    
    if debug == True:
        print "sols",sols;
        print "mut_mask", MutMask_gpu.view();
        print "mut_sols", mutants;

    #return mutated solutions
    return mutants.tolist();
    
""" Helper """
mask_form = ElementwiseKernel(
        "float c, float *x, float *z",
        "z[i] = x[i] <= c ? 1 : 0",
        "if_positive");

crossOver = ElementwiseKernel(
        "int *coPoints, float *a, float *b, float *z",
        "z[i] = i  <= coPoints[i] ? a[i] : b[i]",
        "if_positive");

def form_mutation_mask(MProb_gpu,Mut_mask, mut_prob):
    """takes in the mutation probabilities and forma a mask """
    
    mask_form(mut_prob,MProb_gpu,Mut_mask);
    
def cross_over(SolsA_gpu,SolsB_gpu, coPoint):
    """performs cross over for two solutions """
    
    crossOver(coPoint,SolsA_gpu,SolsB_gpu );
    

    
#TEST - DIFF-EVOLUTION
#L  = numpy.ones((10,10), numpy.float32);
#F = numpy.random.rand(10,10).astype(numpy.float32);
#M = numpy.zeros_like(L).astype(numpy.float32);
#
#mut_list = cuda_diffEvo(L,F,M,0.2);
#print "returned mutants:", mut_list;

#TEST- MUTATION
#Mut = numpy.random.rand(10,10).astype(numpy.float32);
#cuda_mutate(Mut,0.2,0.2,0.0,1.0);

#
##TEST- AGE SOLS
#sols = numpy.random.rand(5,4).astype(numpy.float32);
#cuda_ageSols(sols);