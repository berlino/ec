#require "core" ;;
open Core ;;

(* Types *)

type block = {points : ((int*int)*int) list; original_grid : ((int*int)*int) list} ;;
let example_grid = {points = [(6,4),3; (6,3),2;] ; original_grid = [(6,6),5]} ;;
let example_grid_2 = {points = [(6,4),3; (6,3),2;] ; original_grid = [(6,6),5]} ;;

(* Helper Functions *)

let (--) i j =
  let rec from i j l =
    if i>j then l
    else from i (j-1) (j::l)
    in from i j [] ;;

let (|?) maybe default =
  match maybe with
  | Some v -> v
  | None -> Lazy.force default

let (===) block1 block2 = 
  let block2_val key = (List.Assoc.find block2.points ~equal:(=) key) |? lazy (-1) in
  let block2_has_all_points = List.fold block1.points ~init:true ~f:(fun acc (key,c) -> acc && ((block2_val key) = c)) in
  (block2_has_all_points && ((List.length block1.points) = (List.length block2.points))) ;;
    
(* DSL *)

let move {points;original_grid} x y keep_original = 
  let new_block_points = List.map ~f:(fun ((x_pos,y_pos), color) -> ((x_pos+x, y_pos+x), color)) points in 
  (if keep_original then {points=new_block_points @ original_grid ; original_grid = original_grid} else {points=new_block_points; original_grid = original_grid})
  ;;

let grow {points;original_grid} n = 
  let grow_tile_y ((x_pos,y_pos), color) = List.map ~f:(fun i -> ((x_pos,y_pos+i), color)) (0 -- n) in
  let nested_points = List.map ~f:grow_tile_y points in
  let temp_along_x = List.reduce nested_points ~f:(fun a b -> a @ b) in 
  match temp_along_x with 
  | None -> None
  | Some l -> let grow_tile_x ((x_pos,y_pos), color) = List.map ~f:(fun i -> ((x_pos+i,y_pos), color)) (0 -- n) in
  let along_y_and_x = List.map ~f:grow_tile_x l in
  List.reduce along_y_and_x ~f:(fun a b -> a @ b) ;;


let get_max_y {points;original_grid} = 
  List.fold ~init:0 ~f:(fun curr_sum ((y,x),c)-> (Int.max curr_sum y)) points ;;
let get_max_x {points;original_grid} = 
  List.fold ~init:0 ~f:(fun curr_sum ((y,x),c)-> (Int.max curr_sum x)) points ;;
let get_min_y {points;original_grid} = 
  List.fold ~init:100 ~f:(fun curr_sum ((y,x),c)-> (Int.min curr_sum y)) points ;;
let get_min_x {points;original_grid} = 
  List.fold ~init:100 ~f:(fun curr_sum ((y,x),c)-> (Int.min curr_sum x)) points ;;

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
   {points ; original_grid} ;;  

let print_block block =
  let {points;original_grid} = to_min_grid block true in
    let rec print_points points last_row = 
      match points with 
      | [] -> ();
      | ((y, x),c) :: rest -> if y > last_row then 
        Printf.printf "\n |%i|" c else
        Printf.printf "%i|" c;
        print_points rest y; in
      print_points points (-1);;

let reflect {points;original_grid} is_horizontal = 
  let reflect_point ((y,x),c) = if is_horizontal then ((get_min_y {points;original_grid} - y + get_max_y {points;original_grid} ,x),c)
  else ((y, get_min_x {points;original_grid} - x + get_max_x {points;original_grid}),c) in
  let points = List.map ~f:reflect_point points in 
  {points;original_grid} ;;
  
let merge a b =
  let rec add_until_empty list1 list2 = 
    match list1 with
    | [] -> list2
    | el :: rest -> add_until_empty rest (el :: list2) in
  let points = add_until_empty a.points b.points in
  let original_grid = a.original_grid in
  {points ; original_grid} ;;

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
  [{points = left_half; original_grid = original_grid}; {points = right_half; original_grid = original_grid}] ;;

let is_rectangle block full = 
  (* TODO: Implement non-full version *)
  let {points;original_grid} = to_min_grid block false in
  (List.length points) = (List.length block.points) ;;

let is_symmetrical block is_horizontal = 
  let split_block = split block is_horizontal in
  match split_block with
  | first_half :: second_half -> (reflect first_half is_horizontal) === ((List.nth second_half 0) |? lazy block)
  | [] -> false ;;

is_symmetrical example_grid false ;;




(* let primitive_grow = primitive "grow" (tblock @> tint @> tblock) (grow);;
let primitive_move = primitive "move" (tblock @> tint @> tint @> tblock) (move);;
let primitive_reflect = primitive "reflect" (tblock @> tbool @> tblock) (reflect);;
let primitive_to_min_grid = primitive "blockToMinGrid" (tblock @> tbool @> tgrid) (to_min_grid);;
let primitive_to_block = primitive "gridToBlock" (tgridin @> tblock) ;; *)
