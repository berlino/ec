open Core

(* Types and Helpers*)

let to_grid task = 
  let open Yojson.Basic.Util in
  member "grid" ;;

let (--) i j =
  let rec from i j l =
    if i>j then l
    else from i (j-1) (j::l)
    in from i j [] ;;

let empty_grid height width color = 
let indices = List.cartesian_product (0 -- height) (0 -- width) in
let points = List.map ~f:(fun (y,x) -> ((y,x), color)) indices in
points ;;

type block = {points : ((int*int)*int) list; original_grid : ((int*int)*int) list} ;;
let example_grid = {points = [(0,0),3; (0,1),2; (1,1),2; (4,4),2] ; original_grid = (empty_grid 7 7 0)}
let example_grid_2 = {points = [(6,4),3; (6,3),2;] ; original_grid = [(6,6),5]}

module IntPair = struct
  module T = struct
    type t = int * int
    let compare x y = Tuple2.compare ~cmp1:Int.compare ~cmp2:Int.compare x y
    let sexp_of_t = Tuple2.sexp_of_t Int.sexp_of_t Int.sexp_of_t
    let t_of_sexp = Tuple2.t_of_sexp Int.t_of_sexp Int.t_of_sexp
    let hash = Hashtbl.hash
  end

  include T
  include Comparable.Make(T)
end

let contains item list = 
List.mem list item ~equal:(=)

let (|?) maybe default =
  match maybe with
  | Some v -> v
  | None -> Lazy.force default

let (===) block1 block2 = 
  let block2_val key = (List.Assoc.find block2.points ~equal:(=) key) |? lazy (-1) in
  let block2_has_all_points = List.fold block1.points ~init:true ~f:(fun acc (key,c) -> acc && ((block2_val key) = c)) in
  (block2_has_all_points && ((List.length block1.points) = (List.length block2.points)))

let create_edge_map block colors use_corners = 
  let vertex_points = List.filter block.points ~f:(fun ((y,x),c) -> contains c colors) in
  let vertices = List.map vertex_points ~f:(fun ((y,x),c) -> (y,x)) in
  let basic_adjacent = [(1,0);(0,1);(-1,0);(0,-1)] in
  let adjacent = if use_corners then (basic_adjacent @ [(1,1);(-1,1);(1,-1);(-1,-1)]) else basic_adjacent in
  let vertex_edges (y,x) = List.filter adjacent ~f:(fun (y_inc,x_inc) -> contains (y+y_inc,x+x_inc) vertices) in
  let edge_map = List.map vertices ~f:(fun (y,x) -> ((y,x),(List.map (vertex_edges (y,x)) ~f:(fun (y_edge, x_edge) -> y_edge+y, x_edge+x)))) in
  (* let vertices = List.map (List.filter edge_map ~f:(fun (v,e) -> List.length e > 0)) ~f:(fun (v,e) -> v) in *)
  (vertices, edge_map)

let to_int_pair x = IntPair.Set.choose_exn (IntPair.Set.singleton x) ;;
let to_tuple (a,b) = (a,b) ;;

let dfs graph visited start_node = 
  let rec explore path visited node = 
    (* if (List.mem visited node ~equal:(=)) then raise CycleFound else *)
    if (List.mem path node ~equal:(=)) then visited else     
      let new_path = node :: path in 
      let edges    = (List.Assoc.find_exn graph node ~equal:(=)) in
      let visited  = List.fold_left ~f:(explore new_path) ~init:visited edges in
      node :: visited
  in explore [] visited start_node

(* DSL *)

let get_max_y {points;original_grid} = 
  List.fold ~init:0 ~f:(fun curr_sum ((y,x),c)-> (Int.max curr_sum y)) points
let get_max_x {points;original_grid} = 
  List.fold ~init:0 ~f:(fun curr_sum ((y,x),c)-> (Int.max curr_sum x)) points
let get_min_y {points;original_grid} = 
  List.fold ~init:100 ~f:(fun curr_sum ((y,x),c)-> (Int.min curr_sum y)) points
let get_min_x {points;original_grid} = 
  List.fold ~init:100 ~f:(fun curr_sum ((y,x),c)-> (Int.min curr_sum x)) points

let to_original_grid_overlay {points ; original_grid} with_original = 
  let maxY = get_max_y {points=original_grid;original_grid} in
  let maxX = get_max_x {points=original_grid;original_grid} in
  let indices = List.cartesian_product (0 -- maxY) (0 -- maxX) in
  let deduce_val (y,x) = match List.Assoc.find points (y,x) ~equal:(=) with
      | Some c -> c
      | None -> if with_original then match List.Assoc.find original_grid (y,x) ~equal:(=) with 
        | Some c_original -> c_original
        | None -> 0
      else 0 in
  let points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y,x))) indices in
  {points ; original_grid}

let to_min_grid {points;original_grid} with_original = 
  let minY = get_min_y {points;original_grid} in
  let minX = get_min_x {points;original_grid} in
  let shiftY = (get_max_y {points;original_grid}) - minY in 
  let shiftX = (get_max_x {points;original_grid}) - minX in 
  let indices = List.cartesian_product (0 -- shiftY) (0 -- shiftX) in
  let deduce_val (y,x) = match List.Assoc.find points (y,x) ~equal:(=) with
      | Some c -> c
      | None -> if with_original then match List.Assoc.find original_grid (y,x) ~equal:(=) with 
        | Some c_original -> c_original
        | None -> 0
      else 0 in
  let points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y+minY,x+minX))) indices in
   {points ; original_grid}

let print_block {points ; original_grid}  =
  let maxY = get_max_y {points=original_grid;original_grid} in
  let maxX = get_max_x {points=original_grid;original_grid} in
  let indices = List.cartesian_product (0 -- maxY) (0 -- maxX) in
  let deduce_val (y,x) = match List.Assoc.find points (y,x) ~equal:(=) with
      | Some c -> c
      | None -> (-1) in
  let points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y,x))) indices in
  let rec print_points points last_row = 
    match points with 
    | [] -> Printf.printf "\n\n";
    | ((y, x),c) :: rest -> if y > last_row then 
      (if (c > (-1)) then Printf.printf "\n |%i|" c else  Printf.printf "\n | |") else
      (if (c > (-1)) then Printf.printf "%i|" c else  Printf.printf " |");
      print_points rest y; in
    print_points points (-1)

let print_blocks blocks = List.iter blocks ~f:(fun block -> print_block block);;

let reflect {points;original_grid} is_horizontal = 
  let reflect_point ((y,x),c) = if is_horizontal then ((get_min_y {points;original_grid} - y + get_max_y {points;original_grid} ,x),c)
  else ((y, get_min_x {points;original_grid} - x + get_max_x {points;original_grid}),c) in
  let points = List.map ~f:reflect_point points in 
  {points;original_grid}

let move {points;original_grid} x y keep_original = 
  let new_block_points = List.map ~f:(fun ((x_pos,y_pos), color) -> ((x_pos+x, y_pos+x), color)) points in 
  (if keep_original then {points=new_block_points @ original_grid ; original_grid = original_grid} else {points=new_block_points; original_grid = original_grid})

let grow {points;original_grid} n = 
  let grow_tile_y ((x_pos,y_pos), color) = List.map ~f:(fun i -> ((x_pos,y_pos+i), color)) (0 -- n) in
  let nested_points = List.map ~f:grow_tile_y points in
  let temp_along_x = List.reduce nested_points ~f:(fun a b -> a @ b) in 
  match temp_along_x with 
  | None -> None
  | Some l -> let grow_tile_x ((x_pos,y_pos), color) = List.map ~f:(fun i -> ((x_pos+i,y_pos), color)) (0 -- n) in
  let along_y_and_x = List.map ~f:grow_tile_x l in
  List.reduce along_y_and_x ~f:(fun a b -> a @ b)
  
let merge a b =
  let rec add_until_empty list1 list2 = 
    match list1 with
    | [] -> list2
    | el :: rest -> add_until_empty rest (el :: list2) in
  let points = add_until_empty a.points b.points in
  let original_grid = a.original_grid in
  {points ; original_grid} ;;

let is_rectangle block full = 
  (* TODO: Implement non-full version *)
  let {points;original_grid} = to_min_grid block false in
  (List.length points) = (List.length block.points)

let split block is_horizontal =
  let {points ; original_grid} = block in
  match is_horizontal with 
  | true ->
  let horizontal_length = (get_max_y block) - (get_min_y block) + 1 in
  let halfway = (get_min_y block) + (horizontal_length / 2) in 
  let top_half = List.filter points ~f: (fun ((y,x),c) -> y < halfway) in
  let bottom_half_start = if ((horizontal_length mod 2) = 1) then halfway + 1 else halfway in
  let bottom_half = List.filter points ~f: (fun ((y,x),c) -> y >= bottom_half_start) in
  [{points = top_half; original_grid = original_grid}; {points = bottom_half; original_grid = original_grid}]
  | false ->
  let vertical_length = (get_max_x block) - (get_min_x block) + 1 in
  let halfway = (get_min_x block) + (vertical_length / 2) in 
  let left_half = List.filter points ~f: (fun ((y,x),c) -> x < halfway) in
  let right_half_start = if ((vertical_length mod 2) = 1) then halfway + 1 else halfway in
  let right_half = List.filter points ~f: (fun ((y,x),c) -> x >= right_half_start) in
  [{points = left_half; original_grid = original_grid}; {points = right_half; original_grid = original_grid}]

let is_symmetrical block is_horizontal = 
  let split_block = split block is_horizontal in
  match split_block with
  | first_half :: second_half -> (reflect first_half is_horizontal) === ((List.nth second_half 0) |? lazy block)
  | [] -> false 

let box_block {points;original_grid} = 
  let minY = get_min_y {points;original_grid} in
  let maxY = get_max_y {points;original_grid} in
  let minX = get_min_x {points;original_grid} in
  let maxX = get_max_x {points;original_grid} in
  let indices = List.cartesian_product (minY -- maxY) (minX -- maxX) in
  let deduce_val (y,x) = match List.Assoc.find points (y,x) ~equal:(=) with
      | Some c -> c
      | None -> match List.Assoc.find original_grid (y,x) ~equal:(=) with 
        | Some c_original -> c_original
        | None -> 0 in
  let points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y,x))) indices in
   {points ; original_grid}

let find_blocks_by block colors is_corner box_blocks = 
  let vertices, graph = create_edge_map block colors is_corner in 
  let explore_v state v = if IntPair.Set.mem (List.reduce_exn state ~f:IntPair.Set.union) (to_int_pair v) 
    then state
  else 
    let connected_component_vertices = IntPair.Set.of_list (List.map (dfs graph [] v) ~f:to_int_pair) in
    state @ [connected_component_vertices] in
  let init = [IntPair.Set.empty] in
  let connected_components = List.fold_left ~f:explore_v ~init:init (List.map vertices ~f:to_int_pair) in
  let connected_components_2d_list = List.map connected_components ~f:(fun cc -> 
    let list_cc = IntPair.Set.to_list cc in
    List.map list_cc ~f:(fun int_pair -> to_tuple int_pair)) in
  let deduce_val key = match (List.Assoc.find block.points key ~equal:(=)) with 
  | None -> 0
  | Some c -> c in
  let with_empty = List.map connected_components_2d_list ~f:(fun cc -> 
    {points = List.map cc ~f:(fun key -> (key, deduce_val key)); original_grid = block.original_grid}) in 
  let final_blocks = List.drop with_empty 1 in
  if box_blocks then List.map final_blocks ~f:box_block else final_blocks ;;

let find_same_color_blocks block is_corner box_blocks = 
  let blocks_by_color = List.map (1 -- 9) ~f:(fun color -> find_blocks_by block [color] is_corner box_blocks) in
  List.concat blocks_by_color ;;

let find_blocks_by_black_b block is_corner box_blocks = 
  find_blocks_by block (1 -- 9) is_corner box_blocks ;;

let fill_color block new_color = 
  let points = List.map block.points ~f:(fun ((y,x),c) -> (y,x),new_color) in
  {points = points ; original_grid = block.original_grid}

let replace_color block old_color new_color = 
  let points = List.map block.points ~f:(fun ((y,x),c) -> (y,x), if (c = old_color) then new_color else c) in
  {points = points ; original_grid = block.original_grid} ;;

let merge_blocks blocks = 
  List.reduce_exn blocks ~f:merge ;;

let python_split x =
  let split = String.split_on_chars ~on:[','] x in 
  let filt_split = List.filter split ~f:(fun x -> x <> "") in
  let y = List.nth_exn filt_split 0 |> int_of_string in
  let x = List.nth_exn filt_split 1 |> int_of_string in
  (y,x)
;;

let to_grid task = 
  let open Yojson.Basic.Util in
  let json = task |> member "grid" |> to_assoc in 
  let grid_points = List.map json ~f:(fun (key, color) -> ((python_split key), (to_int color))) in
  let grid = {points = grid_points ; original_grid = grid_points} in 
  grid ;;

let rec print_list = function 
[] -> ()
| e::l -> printf "%d" e ; print_string " " ; print_list l ;;


let convert_raw_to_block raw = 
  let open Yojson.Basic.Util in

  let length = List.length raw - 1 in
  let indices = List.cartesian_product (0 -- length) (0 -- length) in
  let match_row row x = match List.nth row x with
        | Some c -> c |> to_int
        | None -> (-1) in
  let deduce_val (y,x) = match List.nth raw y with
      | Some row -> match_row (row |> to_list) x 
      | None -> (-1) in
  let new_points = List.map ~f:(fun (y,x) -> ((y,x), deduce_val (y,x))) indices in
  {points = new_points; original_grid = new_points} ;;

let test_example assoc_list program = 
  let open Yojson.Basic.Util in
  let raw_input = List.Assoc.find_exn assoc_list "input" ~equal:(=) |> to_list in
  let raw_expected_output = List.Assoc.find_exn assoc_list "output" ~equal:(=) |> to_list in
  let input = convert_raw_to_block raw_input in
  let expected_output = convert_raw_to_block raw_expected_output in
  let got_output = program input in
  print_block got_output;
  print_block expected_output;
  let matched = got_output === expected_output in
  printf "%B \n" matched ;;

let test_task filename program =
  let open Yojson.Basic.Util in
  let open Yojson.Basic in
  let json = from_file filename in
  let json = json |> member "train" |> to_list in
  let pair_list = List.map json ~f:(fun pair -> pair |> to_assoc) in 
  let examples = List.map pair_list ~f:(fun assoc_list -> test_example assoc_list program) in
  examples ;;


let filename = "/Users/theo/Development/program_induction/ec/arc-data/data/training/67a3c6ac.json" in
test_task filename (fun a -> reflect a false) ;;

(* primitives *)

(* ignore(primitive "black" tcolor 0) ;;
ignore(primitive "blue" tcolor 1) ;;
ignore(primitive "red" tcolor 2) ;;
ignore(primitive "green" tcolor 3) ;;
ignore(primitive "yellow" tcolor 4) ;;
ignore(primitive "grey" tcolor 5) ;;
ignore(primitive "pink" tcolor 6) ;;
ignore(primitive "orange" tcolor 7) ;;
ignore(primitive "teal" tcolor 8) ;;
ignore(primitive "maroon" tcolor 9) ;;

(* tblocks -> tblock *)
ignore(primitive "merge_blocks" (tblocks @> tblock) merge_blocks) ;;
ignore(primitive "filter_blocks" ((tblock @> tboolean) @> (tblocks) @> (tblocks)) (fun f l -> List.filter ~f:f l));;
ignore(primitive "map_blocks" ((tblock @> tboolean) @> (tblocks) @> (tblocks)) (fun f l -> List.map ~f:f l));;


(* tblock -> tblock *)
ignore(primitive "reflect" (tblock @> tboolean @> tblock) reflect) ;;
ignore(primitive "move" (tblock @> tint @> tint @> tboolean @> tblock) move) ;;
ignore(primitive "grow" (tblock @> tint @> tblock) grow) ;;
ignore(primitive "fill_color" (tblock @> tcolor @> tblock) fill_color) ;;
ignore(primitive "replace_color" (tblock @> tcolor @> tcolor @> tblock) replace_color) ;;
ignore(primitive "box_block" (tblock @> tblock) box_block) ;;
(* tblock -> tblocks *)
ignore(primitive "split" (tblock @> tboolean @> tblocks) split) ;;
(* tblock -> tgrid *)
ignore(primitive "to_min_grid" (tblock @> tboolean @> tgrid) to_min_grid) ;;
ignore(primitive "to_original_grid_overlay" (tblock @> tboolean @> tgrid) to_original_grid_overlay) ;;


(* tblock -> tboolean *)
ignore(primitive "is_symmetrical" (tblock @> tboolean @> tboolean) is_symmetrical) ;;
ignore(primitive "is_rectangle" (tblock @> tboolean @> tboolean) to_original_grid_overlay) ;;

(* tgrid -> tblocks *)
ignore(primitive "identity" (tgrid @> tgrid) (fun x -> x)) ;;
ignore(primitive "grid_to_block" (tgrid @> tblock) (fun x -> x)) ;;
ignore(primitive "find_same_color_blocks" (tgrid @> tblocks) find_same_color_blocks) ;;
ignore(primitive "find_blocks_by_black_b" (tgrid @> tboolean @> tboolean @> tblocks) find_blocks_by) ;;

 *)

