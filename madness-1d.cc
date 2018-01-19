#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> // pow
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace std;


enum TASK_IDs {
   TOP_LEVEL_TASK_ID,
   REFINE_TASK_ID,
   PRINT_TASK_ID,
};

enum FieldIDs {
   FID_X;
};


struct Arguments {
   /* level of the node in the binary tree. Root is at level 0 */
   int n;

   /* labeling of the node in the binary tree. Root has the value label = 0 
    * Node with (n, l) has it's left child at (n + 1, 2 * l) and it's right child at (n + 1, 2 * l + 1)
    */
   int l;

   int max_depth;

   // Constructor
   Arguments(int n = 0, int l = 0, int max_depth = 32) : n(n), l(l), max_depth(max_depth) {}
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) {


   int max_depth = 32;
   {
      const InputArgs &command_args = HighLevelRuntime::get_input_args();
      if (command_args.argc > 1) {
         max_depth = atoi(command_args.argv[1]);
      }
   }


   // TODO: to be verified by Wonchan
   Rect<2> tree_rect(Point<2>(0, 0) /*min_n = 0, min_l = 0*/, Point<2>(max_depth, static_cast<int>(pow(2, max_depth)) - 1) /*max_n = max_depth, max_l = (2 ^ max_depth) - 1*/);
   IndexSpace is = runtime->create_index_space(ctx, Domain::from_rect<2>(tree_rect));
   FieldSpace fs = runtime->create_field_space(ctx);
   {
      FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
      allocator.allocate_field(sizeof(int), FID_X);
   }

   LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
   struct Arguments args(0, 0, max_depth); 
   
   // Launching the refine task
   TaskLauncher refine_launcher(REFINE_TASK_ID, TaskArgument(&args, sizeof(struct Arguments)));
   refine_launcher.add_region_requirement(RegionRequirement(lr, WRITE_DISCARD, EXCLUSIVE, lr));
   refine_launcher.add_field(0, FID_X);
   runtime->execute_task(ctx, refine_launcher);

   // Launching another task to print the values of the binary tree nodes
   TaskLauncher print_launcher(PRINT_TASK_ID, TaskArgument(&args, sizeof(struct Arguments)));
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



// To be recursive task calling for the left and right subtrees, if necessary !
void refine_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, HighLevelRuntime *runtime) {


   struct Arguments args = *(const struct Arguments *) task->args;
   int n = args.n;
   int l = args.l;
   int max_depth = args.max_depth;

   int node_value = (rand() % 10) + 1;
   if (node_value <= 3 || n >= max_depth) {
      // TODO: To be completed
      // store in the hash_map (n, l) --> node_value;
   }
   else {
      // TODO: To be completed      
      // store in the hash_map (n, l) --> ZERO
     
      // we have only two subtrees --> Point<1>(0) and Point<1>(1)
      Rect<1> color_bounds(Point<1>(0), Point<1>(1));
      Domain color_domain = Domain::from_rect<1>(color_bounds);

      IndexPartition ip;
      DomainColoring coloring;

      Rect<2> left_sub_tree_rect(Point<2>(n+1, 2 * l), Point<2>(max_depth, static_cast<int>(pow(2, max_depth - 1)) - 1));
      coloring[0] = Domain::from_rect<2>(left_sub_tree_rect);
      
      Rect<2> right_sub_tree_rect(Point<2>(n+1, 2 * l + 1), Point<2>(max_depth, static_cast<int>(pow(2, max_depth)) - 1));
      coloring[1] = Domain::from_rect<2>(right_sub_tree_rect);

      ip = runtime->create_index_partition(ctx, is, color_domain, coloring, true /* disjoint */);
      LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

      // TODO: Here, I can't have IndexLauncher as I need to pass (n+1, 2 *l, max_depth) as an argument to one and pass (n+1, 2 * l + 1, max_depth) to another one ! 


      // make a new task of refine_task with arguments (n+1, 2 * l) --> left subtree
      // make a new task of refine_task with arguments (n+1, 2 * l + 1) --> right subtree
   }
}

