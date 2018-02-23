#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> // pow
#include "legion.h"

using namespace Legion;
using namespace std;

enum TASK_IDs {
   TOP_LEVEL_TASK_ID,
   REFINE_TASK_ID,
   SET_TASK_ID,
   PRINT_TASK_ID,
};

enum FieldIDs {
   FID_X,
};

struct Arguments {
   /* level of the node in the binary tree. Root is at level 0 */
   int n;

   /* labeling of the node in the binary tree. Root has the value label = 0 
    * Node with (n, l) has it's left child at (n + 1, 2 * l) and it's right child at (n + 1, 2 * l + 1)
    */
   int l;

   int max_depth;

   coord_t idx;

   drand48_data gen;

   // Constructor
   Arguments(int _n = 0, int _l = 0, int _max_depth = 32, coord_t _idx = 0LL)
     : n(_n), l(_l), max_depth(_max_depth), idx(_idx)
   {}
};

struct SetTaskArgs {
  int node_value;
  coord_t idx;
  SetTaskArgs(int _node_value, coord_t _idx) : node_value(_node_value), idx(_idx) {}
};

//   k=1 (1 subregion per node)
//                0
//         1             8
//     2      5      9      12
//   3   4  6   7  10  11  13   14
//
//       i              (n, l)
//    il    ir   (n+1, 2*l)  (n+1, 2*l+1)
//
//    il = i + 1
//    ir = i + 2^(max_level -l)
//
//    when each subtree holds k levels
//    [i .. i+(2^k-1)-1]
//    0 <= j <= 2^k-1 => [i+(2^k-1)-1 + 1 +  j      * (2^(max_level - (l + k) +1) - 1) ..
//                        i+(2^k-1)-1 + 1 + (j + 1) * (2^(max_level - (l + k) +1) - 1) - 1]
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) {


   int max_depth = 4;
   long int seed = 12345;
   {
      const InputArgs &command_args = HighLevelRuntime::get_input_args();
      for (int idx = 1; idx < command_args.argc; ++idx)
      {
        if (strcmp(command_args.argv[idx], "-max_depth") == 0)
          max_depth = atoi(command_args.argv[++idx]);
        else if (strcmp(command_args.argv[idx], "-seed") == 0)
          seed = atol(command_args.argv[++idx]);
      }
   }

   Rect<1> tree_rect(0LL, static_cast<coord_t>(pow(2, max_depth + 1)) - 2);
   IndexSpace is = runtime->create_index_space(ctx, tree_rect);
   FieldSpace fs = runtime->create_field_space(ctx);
   {
      FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
      allocator.allocate_field(sizeof(int), FID_X);
   }

   LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
   Arguments args(0, 0, max_depth, 0);
   srand48_r(seed, &args.gen);

   // Launching the refine task
   TaskLauncher refine_launcher(REFINE_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
   refine_launcher.add_region_requirement(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
   refine_launcher.add_field(0, FID_X);
   runtime->execute_task(ctx, refine_launcher);

   // Launching another task to print the values of the binary tree nodes
   TaskLauncher print_launcher(PRINT_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
   print_launcher.add_region_requirement(RegionRequirement(lr, READ_ONLY, EXCLUSIVE, lr));
   print_launcher.add_field(0, FID_X);
   runtime->execute_task(ctx, print_launcher);


   // Destroying allocated memory

   runtime->destroy_logical_region(ctx, lr);
   runtime->destroy_field_space(ctx, fs);
   runtime->destroy_index_space(ctx, is);
}


/*
 *
 *  This algorithm generates a binary tree (and only leaves contain the valuable data). Initial call would be Refine(0,0):
 *   
 *  1) Refine(int n, int l) {
 *  2)        int node_value = pick a random value in a range [1, 10], inclusive;
 *
 *  3)        if (node_value <= 3 || n >= MAX_DEPTH) {
 *  4)                   store in the hash_map (n, l) --> node_value;
 *  5)        }
 *  6)        else {
 *  7)                   store in the hash_map (n, l) --> ZERO; // ZERO value indicates that the node is an internal node
 *  8)                   make a new task of Refine with arguments(n+1, 2 * l); // left child
 *  9)                   make a new task of Refine with arguments(n+1, 2 * l + 1); // right child
 *  10)      }
 *  11) }
 *
 *  
 *
 *  So, as you can clearly see, the result of this task is a binary tree, whose internal nodes contain the value ZERO and only it's leaves contain values in range [1, 3]. As an example, the following could be the result of running the ALG-1:
 *
 *                        _____________0_____________                                 DEPTH/LEVEL = 0
 *                  _____0____                 ______0_______                         DEPTH/LEVEL = 1
 *             ____0___       1            ___0___         __0____                    DEPTH/LEVEL = 2
 *            2        1                  3     __0__     1     __0__                 DEPTH/LEVEL = 3
 *                                           __0__   3         1     2                DEPTH/LEVEL = 4
 *                                          2     2                                   DEPTH/LEVEL = 5
 *
 *
 *  This tree is called to be in "scaling or refined form".
 *
 * */


void set_task(const Task *task,
              const std::vector<PhysicalRegion> &regions,
              Context ctx, HighLevelRuntime *runtime) {

  SetTaskArgs args = *(const SetTaskArgs *) task->args;
  assert(regions.size() == 1);
  const FieldAccessor<WRITE_DISCARD,int,1> write_acc(regions[0], FID_X);
  if (args.node_value <= 3) {
    write_acc[args.idx] = args.node_value;
  }
  else {
    write_acc[args.idx] = 0;
  }
}

// To be recursive task calling for the left and right subtrees, if necessary !
void refine_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, HighLevelRuntime *runtime) {

   Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
                                         : *(const Arguments *) task->args;
   int n = args.n;
   int l = args.l;
   int max_depth = args.max_depth;
   coord_t idx = args.idx;

   assert(regions.size() == 1);
   LogicalRegion lr = regions[0].get_logical_region();
   LogicalPartition lp = LogicalPartition::NO_PART;
   LogicalRegion my_sub_tree_lr = lr;

   DomainPoint my_sub_tree_color(Point<1>(0LL));
   DomainPoint left_sub_tree_color(Point<1>(1LL));
   DomainPoint right_sub_tree_color(Point<1>(2LL));
   coord_t idx_left_sub_tree = 0LL;
   coord_t idx_right_sub_tree = 0LL;

   if (n < max_depth)
   {
     IndexSpace is = lr.get_index_space();
     DomainPointColoring coloring;

     idx_left_sub_tree = idx + 1;
     idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

     Rect<1> my_sub_tree_rect(idx, idx);
     Rect<1> left_sub_tree_rect(idx_left_sub_tree, idx_right_sub_tree - 1);
     Rect<1> right_sub_tree_rect(idx_right_sub_tree,
                                 idx_right_sub_tree + static_cast<coord_t>(pow(2, max_depth - n)) - 2);
     /*
     fprintf(stderr, "(n: %d, l: %d) - idx: [%lld, %lld] (max_depth: %d)\n"
                     "  |-- (n: %d, l: %d) - idx: [%lld, %lld] (max_depth: %d)\n"
                     "  |-- (n: %d, l: %d) - idx: [%lld, %lld] (max_depth: %d)\n",
         n, l, idx, idx, max_depth,
         n + 1, 2 * l,     left_sub_tree_rect.lo[0],  left_sub_tree_rect.hi[0],  max_depth,
         n + 1, 2 * l + 1, right_sub_tree_rect.lo[0], right_sub_tree_rect.hi[0], max_depth); */

     coloring[my_sub_tree_color] = my_sub_tree_rect;
     coloring[left_sub_tree_color] = left_sub_tree_rect;
     coloring[right_sub_tree_color] = right_sub_tree_rect;

     Rect<1> color_space = Rect<1>(my_sub_tree_color, right_sub_tree_color);

     IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND);
     lp = runtime->get_logical_partition(ctx, lr, ip);
     my_sub_tree_lr = runtime->get_logical_subregion_by_color(ctx, lp, my_sub_tree_color);
   }
   assert(lr != LogicalRegion::NO_REGION);
   assert(my_sub_tree_lr != LogicalRegion::NO_REGION);

   long int node_value;
   lrand48_r(&args.gen, &node_value);
   node_value = node_value % 10 + 1;
   {
     SetTaskArgs args(node_value, idx);
     TaskLauncher set_task_launcher(SET_TASK_ID, TaskArgument(&args, sizeof(SetTaskArgs)));
     RegionRequirement req(my_sub_tree_lr, WRITE_DISCARD, EXCLUSIVE, lr);
     req.add_field(FID_X);
     set_task_launcher.add_region_requirement(req);
     runtime->execute_task(ctx, set_task_launcher);
   }

   if (node_value > 3 && n < max_depth)
   {
     assert(lp != LogicalPartition::NO_PART);
     Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
     ArgumentMap arg_map;
     Arguments for_left_sub_tree (n + 1, l * 2    , max_depth, idx_left_sub_tree);
     Arguments for_right_sub_tree(n + 1, l * 2 + 1, max_depth, idx_right_sub_tree);

     // Make sure two subtrees use different random number generators
     long int new_seed = 0L;
     lrand48_r(&args.gen, &new_seed);
     for_left_sub_tree.gen = args.gen;
     srand48_r(new_seed, &for_right_sub_tree.gen);

     arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(Arguments)));
     arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(Arguments)));

     IndexTaskLauncher refine_launcher(REFINE_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
     RegionRequirement req(lp, 0, WRITE_DISCARD, EXCLUSIVE, lr);
     req.add_field(FID_X);
     refine_launcher.add_region_requirement(req);
     runtime->execute_index_space(ctx, refine_launcher);
   }
}

void print_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                Context ctxt, HighLevelRuntime *runtime) {

   Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
                                         : *(const Arguments *) task->args;

   int n = args.n,
       l = args.l,
       max_depth = args.max_depth;

   coord_t idx = args.idx;

   const FieldAccessor<READ_ONLY, int, 1> read_acc(regions[0], FID_X);
   int node_value = read_acc[idx];

   fprintf(stderr, "(n: %d, l: %d), idx: %lld, node_value: %d\n", n, l, idx, node_value);

   if (node_value == 0) { // The current node is an internal node; launching two sub-task for the left and right subtrees
      LogicalRegion lr = regions[0].get_logical_region();
      LogicalPartition lp = LogicalPartition::NO_PART;

      DomainPoint left_sub_tree_color(Point<1>(0LL));
      DomainPoint right_sub_tree_color(Point<1>(1LL));

      IndexSpace is = lr.get_index_space();
      DomainPointColoring coloring;

      coord_t idx_left_sub_tree = idx + 1;
      coord_t idx_right_sub_tree = idx + static_cast<coord_t>(pow(2, max_depth - n));

      Rect<1> left_sub_tree_rect(idx_left_sub_tree, idx_right_sub_tree - 1);
      Rect<1> right_sub_tree_rect(idx_right_sub_tree,
                                  idx_right_sub_tree + static_cast<coord_t>(pow(2, max_depth - n)) - 2);

      coloring[left_sub_tree_color] = left_sub_tree_rect;
      coloring[right_sub_tree_color] = right_sub_tree_rect;

      Rect<1> color_space = Rect<1>(left_sub_tree_color, right_sub_tree_color);
      IndexPartition ip = runtime->create_index_partition(ctxt, is, color_space, coloring, DISJOINT_KIND);
      lp = runtime->get_logical_partition(ctxt, lr, ip);

      Rect<1> launch_domain(left_sub_tree_color, right_sub_tree_color);
      ArgumentMap arg_map;
      
      Arguments for_left_sub_tree(n+1, 2 * l, max_depth, idx_left_sub_tree);
      Arguments for_right_sub_tree(n+1, 2 * l + 1, max_depth, idx_right_sub_tree);
      
      arg_map.set_point(left_sub_tree_color, TaskArgument(&for_left_sub_tree, sizeof(Arguments)));
      arg_map.set_point(right_sub_tree_color, TaskArgument(&for_right_sub_tree, sizeof(Arguments)));

      IndexTaskLauncher print_launcher(PRINT_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
      RegionRequirement req(lp, 0, READ_ONLY, EXCLUSIVE, lr);
      req.add_field(FID_X);

      print_launcher.add_region_requirement(req);
      runtime->execute_index_space(ctxt, print_launcher);
   }
   
}


int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(REFINE_TASK_ID, "refine");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner(true);
    Runtime::preregister_task_variant<refine_task>(registrar, "refine");
  }

  {
    TaskVariantRegistrar registrar(SET_TASK_ID, "set");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<set_task>(registrar, "set");
  }

  {
    TaskVariantRegistrar registrar(PRINT_TASK_ID, "print");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<print_task>(registrar, "print");
  }


  return Runtime::start(argc, argv);
}
