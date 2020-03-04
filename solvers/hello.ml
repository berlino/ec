#require "core" ;;
open Core ;;
open Type ;;

(* Types *)

type block = {points : ((int*int)*int) list; original_grid : ((int*int)*int) list} ;;
let example_grid = {points = [(7,5),3; (7,7),9;] ; original_grid = [(6,6),5]} ;;
let example_grid_2 = {points = [(6,4),3; (7,7),3;] ; original_grid = [(6,6),5]} ;;

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


let getMaxY {points;original_grid} = 
  List.fold ~init:0 ~f:(fun curr_sum ((y,x),c)-> (Int.max curr_sum y)) points ;;
let getMaxX {points;original_grid} = 
  List.fold ~init:0 ~f:(fun curr_sum ((y,x),c)-> (Int.max curr_sum x)) points ;;
let getMinY {points;original_grid} = 
  List.fold ~init:100 ~f:(fun curr_sum ((y,x),c)-> (Int.min curr_sum y)) points ;;
let getMinX {points;original_grid} = 
  List.fold ~init:100 ~f:(fun curr_sum ((y,x),c)-> (Int.min curr_sum x)) points ;;

let to_min_grid {points;original_grid} with_original = 
  let minY = getMinY {points;original_grid} in
  let minX = getMinX {points;original_grid} in
  let shiftY = (getMaxY {points;original_grid}) - minY in 
  let shiftX = (getMaxX {points;original_grid}) - minX in 
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
  let {points;original_grid} = to_min_grid block 0 true in
    let rec print_points points last_row = 
      match points with 
      | [] -> ();
      | ((y, x),c) :: rest -> if y > last_row then 
        Printf.printf "\n |%i|" c else
        Printf.printf "%i|" c;
        print_points rest y; in
      print_points points (-1);;

let reflect {points;original_grid} is_horizontal = 
  let reflect_point ((y,x),c) = if is_horizontal then ((getMinY {points;original_grid} - y + getMaxY {points;original_grid} ,x),c)
  else ((y, getMinX {points;original_grid} - x + getMaxX {points;original_grid}),c) in
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

print_block (merge example_grid example_grid_2);;


let primitive_grow = primitive "grow" (tblock @> tint @> tblock) (grow);;
let primitive_move = primitive "move" (tblock @> tint @> tint @> tblock) (move);;
let primitive_reflect = primitive "reflect" (tblock @> tbool @> tblock) (reflect);;
let primitive_merge = primitive "merge" (tblock @> tblock @> tblock) (merge);;
let primitive_to_min_grid = primitive "blockToMinGrid" (tblock @> tbool @> tgrid) (to_min_grid);;

let primitive_to_block = primitive "gridToBlock" (tgridin @> tblock) ;;
