

type block = {points : ((int*int)*int) list; original_grid : ((int*int)*int) list} ;;
let example_grid = {points = [(0,1),3; (0,2),3;] ; original_grid = [(0,1),5; (4,1),3;]} ;;

(* def move(self, y, x, keepOriginal=False):
      newPoints = self.points.copy() if keepOriginal else {}
      for yPos,xPos in self.points.keys():
          color = self.points[(yPos,xPos)]
          newPoints[yPos + y, xPos + x] = color
      return self.fromPoints(newPoints) *)

let move {points;original_grid} x y keep_original = 
  let new_block_points = List.map (fun ((x_pos,y_pos), color) -> ((x_pos+x, y_pos+x), color)) points in 
  (if keep_original then {points=new_block_points @ original_grid ; original_grid = original_grid} else {points=new_block_points; original_grid = original_grid})
  ;;

let grow {points;original_grid} n = 
  let rec make_list n = match n with 
  -1 -> [] | 
  _ -> (n :: make_list (n-1)) in
  let grow_tile ((x_pos,y_pos), color) = List.map (fun i -> ((x_pos+i,y_pos+i), color)) (make_list n) in
  let nested_points = List.map grow_tile points in
  List.reduce ~f:(fun (a,b) -> a @ b) nested_points
  ;;

(* grow example_grid 3;; *)
