open Arc
open Core
open Program
open Timeout
open Type
open Utils

open Yojson.Basic

exception EnumerationTimeout

type task =
  { name: string; task_type: tp;
    log_likelihood: program -> float;
  }

let arc_handler = (fun extras ?timeout:(timeout = 0.001) name ty examples ->
(* Printf.eprintf "Making an arc task %s \n" name; *)
{ name = name    ;
    task_type = ty ;
    log_likelihood =
      (fun p -> 
        flush_everything () ;
        let p = analyze_lazy_evaluation p in
        (* Printf.eprintf "\nFinished analyze_lazy_evaluation"; *)
        flush_everything();
        let rec loop = function
          | [] -> true
          | (xs,y) :: e ->
            try
              match run_for_interval
                      timeout
                      (fun () ->
                        let expected = magical y in
                        flush_everything();
                        let got = magical (run_lazy_analyzed_with_arguments p xs) in
                      got === expected)
              with
                | Some(true) -> loop e
                | _ -> false
            with (* We have to be a bit careful with exceptions if the
                  * synthesized program generated an exception, then we just
                  * terminate w/ false but if the enumeration timeout was
                  * triggered during program evaluation, we need to pass the
                  * exception on
                  *)
              | UnknownPrimitive(n) -> raise (Failure ("Unknown primitive: "^n))
              | EnumerationTimeout  -> raise EnumerationTimeout
              | _                   -> false
        in
        if loop examples
          then 0.0
          else log 0.0)
}) ;;

let load_problems channel =
  let open Yojson.Basic.Util in
  let j = Yojson.Basic.from_channel channel in
  let timeout = try
      j |> member "programTimeout" |> to_float
    with _ ->
      begin
        let defaultTimeout = 0.1 in
        Printf.eprintf
          "\t(ocaml) WARNING: programTimeout not set. Defaulting to %f.\n"
          defaultTimeout ;
        defaultTimeout
      end
  in
  let rec unpack x =
    let open Yojson.Basic in
    to_file "task.txt" x;
    try magical (x |> to_int) with _ ->
    try magical (x |> to_float) with _ ->
    try magical (x |> to_bool) with _ ->
    try magical (x |> to_grid) with _ ->
    try
      let v = x |> to_string in
      if String.length v = 1 then magical v.[0] else magical v
    with _ ->
    try
      x |> to_list |> List.map ~f:unpack |> magical
    with _ -> raise (Failure "could not unpack")
  in

  let tf = j |> member "tasks" |> to_list |> List.map ~f:(fun j -> 
      let e = j |> member "examples" |> to_list in
      let task_type = j |> member "request" |> deserialize_type in 
      let examples = e |> List.map ~f:(fun ex -> (ex |> member "inputs" |> to_list |> List.map ~f:unpack,
                                                  ex |> member "output" |> unpack)) in
      let name = j |> member "name" |> to_string in

      let task = arc_handler (j |> member "extras") ~timeout:timeout name task_type examples
      in
      let programs = j |> member "programs" |> to_list |> List.map ~f:(fun p_json -> p_json |> to_string |> parse_program |> get_some) in
      (* List.iter ~f:(fun p -> Printf.eprintf "%s " (string_of_program p)) programs; *)
      let log_likelihoods = programs |> List.map ~f:(fun p ->
        try
          task.log_likelihood p
        with
          | e -> Printf.eprintf "error"; -1.0
      ) in
      (* List.iter ~f:(fun ll -> Printf.eprintf "%f " ll) log_likelihoods; *)
      (task, programs, log_likelihoods))
  in
  tf;;

let output_job result =
  let open Yojson.Basic.Util in
  let message = 
    `List(result |> List.map ~f:(fun (task, programs, log_likelihoods) ->
        `Assoc(["task", `String(task.name);
                "log_likelihoods", `List(log_likelihoods |> List.map ~f:(fun ll -> `Float(ll)))])))
  in 
  message
;;

load_problems Pervasives.stdin |> output_job |> to_channel Pervasives.stdout;;

(*
let parsing_test_cases() =
  parsing_test_case "(lambda (blocks_to_min_grid (filter_blocks (find_same_color_blocks $0 true false) (lambda (is_tile $0))) false false))";
  parsing_test_case "(lambda (to_min_grid (grow (merge_blocks (find_blocks_by_black_b $0 true false) true) 1) false))"
 in 
parsing_test_cases();;
*)
