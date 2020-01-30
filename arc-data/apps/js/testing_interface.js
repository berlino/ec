
// Internal state.
var CURRENT_INPUT_GRID = new Grid(3, 3);
var CURRENT_OUTPUT_GRID = new Grid(3, 3);
var TEST_PAIRS = new Array();
var CURRENT_TEST_PAIR_INDEX = 0;
var COPY_PASTE_DATA = new Array();
var TASK_FROM_LIST_COUNT = 0

// Cosmetic.
var EDITION_GRID_HEIGHT = 500;
var EDITION_GRID_WIDTH = 500;
var MAX_CELL_SIZE = 100;

function resetTask() {
    CURRENT_INPUT_GRID = new Grid(3, 3);
    TEST_PAIRS = new Array();
    CURRENT_TEST_PAIR_INDEX = 0;
    $('#task_preview').html('');
    resetOutputGrid();
}

function refreshEditionGrid(jqGrid, dataGrid) {
    fillJqGridWithData(jqGrid, dataGrid);
    setUpEditionGridListeners(jqGrid);
    fitCellsToContainer(jqGrid, dataGrid.height, dataGrid.width, EDITION_GRID_HEIGHT, EDITION_GRID_HEIGHT);
    initializeSelectable();
}

function syncFromEditionGridToDataGrid() {
    copyJqGridToDataGrid($('#output_grid .edition_grid'), CURRENT_OUTPUT_GRID);
}

function syncFromDataGridToEditionGrid() {
    refreshEditionGrid($('#output_grid .edition_grid'), CURRENT_OUTPUT_GRID);
}

function getSelectedSymbol() {
    selected = $('#symbol_picker .selected-symbol-preview')[0];
    return $(selected).attr('symbol');
}

function setUpEditionGridListeners(jqGrid) {
    jqGrid.find('.cell').click(function(event) {
        cell = $(event.target);
        symbol = getSelectedSymbol();

        mode = $('input[name=tool_switching]:checked').val();
        if (mode == 'floodfill') {
            // If floodfill: fill all connected cells.
            syncFromEditionGridToDataGrid();
            grid = CURRENT_OUTPUT_GRID.grid;
            floodfillFromLocation(grid, cell.attr('x'), cell.attr('y'), symbol);
            syncFromDataGridToEditionGrid();
        }
        else if (mode == 'edit') {
            // Else: fill just this cell.
            setCellSymbol(cell, symbol);
        }
    });
}

function resizeOutputGrid() {
    size = $('#output_grid_size').val();
    size = parseSizeTuple(size);
    height = size[0];
    width = size[1];

    jqGrid = $('#output_grid .edition_grid');
    syncFromEditionGridToDataGrid();
    dataGrid = JSON.parse(JSON.stringify(CURRENT_OUTPUT_GRID.grid));
    CURRENT_OUTPUT_GRID = new Grid(height, width, dataGrid);
    refreshEditionGrid(jqGrid, CURRENT_OUTPUT_GRID);
}

function resetOutputGrid() {
    syncFromEditionGridToDataGrid();
    CURRENT_OUTPUT_GRID = new Grid(3, 3);
    syncFromDataGridToEditionGrid();
    resizeOutputGrid();
}

function copyFromInput() {
    syncFromEditionGridToDataGrid();
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(CURRENT_INPUT_GRID.grid);
    syncFromDataGridToEditionGrid();
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
}

function fillPairPreview(pairId, inputGrid, outputGrid) {
    var pairSlot = $('#pair_preview_' + pairId);
    if (!pairSlot.length) {
        // Create HTML for pair.
        pairSlot = $('<div id="pair_preview_' + pairId + '" class="pair_preview" index="' + pairId + '"></div>');
        pairSlot.appendTo('#task_preview');
    }
    var jqInputGrid = pairSlot.find('.input_preview');
    if (!jqInputGrid.length) {
        jqInputGrid = $('<div class="input_preview"></div>');
        jqInputGrid.appendTo(pairSlot);
    }
    var jqOutputGrid = pairSlot.find('.output_preview');
    if (!jqOutputGrid.length) {
        jqOutputGrid = $('<div class="output_preview"></div>');
        jqOutputGrid.appendTo(pairSlot);
    }

    fillJqGridWithData(jqInputGrid, inputGrid);
    fitCellsToContainer(jqInputGrid, inputGrid.height, inputGrid.width, 200, 200);
    fillJqGridWithData(jqOutputGrid, outputGrid);
    fitCellsToContainer(jqOutputGrid, outputGrid.height, outputGrid.width, 200, 200);
}

function loadJSONTask(train, test) {
    resetTask();
    $('#modal_bg').hide();
    $('#error_display').hide();
    $('#info_display').hide();

    for (var i = 0; i < train.length; i++) {
        pair = train[i];
        values = pair['input'];
        input_grid = convertSerializedGridToGridObject(values)
        values = pair['output'];
        output_grid = convertSerializedGridToGridObject(values)
        fillPairPreview(i, input_grid, output_grid);
    }
    for (var i=0; i < test.length; i++) {
        pair = test[i];
        TEST_PAIRS.push(pair);
    }
    values = TEST_PAIRS[0]['input'];
    CURRENT_INPUT_GRID = convertSerializedGridToGridObject(values)
    fillTestInput(CURRENT_INPUT_GRID);
    CURRENT_TEST_PAIR_INDEX = 0;
    $('#current_test_input_id_display').html('1');
    $('#total_test_input_count_display').html(test.length);
}

function loadTaskFromFile(e) {
    var file = e.target.files[0];
    if (!file) {
        errorMsg('No file selected');
        return;
    }
    var reader = new FileReader();
    reader.onload = function(e) {
        var contents = e.target.result;

        try {
            contents = JSON.parse(contents);
            train = contents['train'];
            test = contents['test'];
        } catch (e) {
            errorMsg('Bad file format');
            return;
        }
        loadJSONTask(train, test);
    };
    reader.readAsText(file);
    document.getElementById('taskName').innerHTML = file.name;
    document.getElementById('random_task').innerHTML = '';
    document.getElementById('task_from_list').innerHTML = '';
}

function taskFromList() {
    console.log(TASK_FROM_LIST_COUNT)
    var taskList = 
['6d0aefbc.json',
'97999447.json',
'ce4f8723.json',
'c9e6f938.json',
'72ca375d.json',
'0520fde7.json',
'5521c0d9.json',
'007bbfb7.json',
'c59eb873.json',
'fcb5c309.json',
'f25fbde4.json',
'67e8384a.json',
'50cb2852.json',
'6150a2bd.json',
'68b16354.json',
'8f2ea7aa.json',
'6fa7a44f.json',
'3af2c5a8.json',
'62c24649.json',
'4347f46a.json',
'3c9b0459.json',
'4c4377d9.json',
'67a3c6ac.json',
'a416b8f3.json',
'1cf80156.json',
'bb43febb.json',
'9172f3a0.json',
'8be77c9e.json',
]

var programList = 
[
'(lambda (solvec9e6f938 $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000',
'(lambda (solve97999447 $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000',
'(lambda (_solvece4f8723 $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000',
'(lambda (solvec9e6f938 $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000',
'(lambda (_solve72ca375d $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000',
'(lambda (solve0520fde7 $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000',
'(lambda (_solve5521c0d9 $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000',
'(lambda (solve007bbfb7 $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000',
'(lambda (solvef25fbde4 $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000',
'(lambda (solvefcb5c309 $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000',
'(lambda (solvef25fbde4 $0)) <br> <br> log prior = -6.591674 <br> log likelihood = 0.000000', 
'(lambda (concatNAndReflect (solvec9e6f938 $0) true down)) <br> <br> log prior = -12.660099 <br> log likelihood = 0.000000',
'(lambda (solve50cb2852 $0 teal)) <br> <br> log prior = -9.230731 <br> log likelihood = 0.000000',
'(lambda (reflect (reflect $0 false) true)) <br> <br> log prior = -12.660099 <br> log likelihood = 0.000000',
'(lambda (reflect $0 true)) <br> <br> log prior = -7.977968 <br> log likelihood = 0.000000',
'(lambda (solve007bbfb7 (blocksToMinGrid (findBlocksByCorner $0)))) <br> <br> log prior = -12.190096 <br> log likelihood = 0.000000',
'(lambda (concatNAndReflect $0 true down)) <br> <br> log prior = -9.364262 <br> log likelihood = 0.000000',
'(lambda (concatNAndReflect (solvec9e6f938 $0) true down)) <br> <br> log prior = -12.660099 <br> log likelihood = 0.000000',
'(lambda (concatNAndReflect (solvec9e6f938 $0) true down)) <br> <br> log prior = -12.660099 <br> log likelihood = 0.000000',
'(lambda (solve50cb2852 $0 black)) <br> <br> log prior = -9.230731 <br> log likelihood = 0.000000',
'(lambda (reflect (reflect $0 false) true)) <br> <br> log prior = -12.660099 <br> log likelihood = 0.000000',
'(lambda (concatNAndReflect $0 true up)) <br> <br> log prior = -9.364262 <br> log likelihood = 0.000000',
'(lambda (reflect $0 false)) <br> <br> log prior = -7.977968 <br> log likelihood = 0.000000',
'(lambda (duplicateN $0 left 1)) <br> <br> log prior = -10.922407 <br> log likelihood = 0.000000',
'(lambda (blocksToMinGrid (findBlocksByCorner $0))) <br> <br> log prior = -8.894259 <br> log likelihood = 0.000000',
'(lambda (solve50cb2852 $0 red)) <br> <br> log prior = -9.230731 <br> log likelihood = 0.000000',
'(lambda (grow $0 3)) <br> <br> log prior = -9.536113 <br> log likelihood = 0.000000',
'(lambda (concatNAndReflect $0 true down)) <br> <br> log prior = -9.364262 <br> log likelihood = 0.000000',
]

    document.getElementById('task_from_list').innerHTML = (TASK_FROM_LIST_COUNT+1).toString() + ' out of ' + taskList.length.toString() + ' tasks solved: ' + taskList[TASK_FROM_LIST_COUNT];
    document.getElementById('program_found').innerHTML = programList[TASK_FROM_LIST_COUNT]
    document.getElementById('random_task').innerHTML = '';
    var subset = "training";
    $.getJSON("https://api.github.com/repos/fchollet/ARC/contents/data/" + subset, function(tasks) {
      var task = tasks.find(task => task.name == taskList[TASK_FROM_LIST_COUNT])
      TASK_FROM_LIST_COUNT = (TASK_FROM_LIST_COUNT + 1) % taskList.length
      $.getJSON(task["download_url"], function(json) {
          try {
              train = json['train'];
              test = json['test'];
          } catch (e) {
              errorMsg('Bad file format');
              return;
          }
          loadJSONTask(train, test);
          $('#load_task_file_input')[0].value = "";
          infoMsg("Loaded task training/" + task["name"]);
      })
      .error(function(){
        errorMsg('Error loading task');
      });
        document.getElementById('taskName').innerHTML = task['name'];

    })
    .error(function(){
      errorMsg('Error loading task list');
    });
}

function randomTask() {
    TASK_FROM_LIST_COUNT = 0
    var subset = "training";
    $.getJSON("https://api.github.com/repos/fchollet/ARC/contents/data/" + subset, function(tasks) {
      var task = tasks.find(task => task.name == 'fcb5c309.json')
      var task = tasks[Math.floor(Math.random() * tasks.length)];
      $.getJSON(task["download_url"], function(json) {
          try {
              train = json['train'];
              test = json['test'];
          } catch (e) {
              errorMsg('Bad file format');
              return;
          }
          loadJSONTask(train, test);
          $('#load_task_file_input')[0].value = "";
          infoMsg("Loaded task training/" + task["name"]);
      })
      .error(function(){
        errorMsg('Error loading task');
      });
        document.getElementById('taskName').innerHTML = task['name'];
        document.getElementById('random_task').innerHTML = task['name'];
        document.getElementById('task_from_list').innerHTML = '';
    })
    .error(function(){
      errorMsg('Error loading task list');
    });
}

function nextTestInput() {
    if (TEST_PAIRS.length <= CURRENT_TEST_PAIR_INDEX + 1) {
        errorMsg('No next test input. Pick another file?')
        return
    }
    CURRENT_TEST_PAIR_INDEX += 1;
    values = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['input'];
    CURRENT_INPUT_GRID = convertSerializedGridToGridObject(values)
    fillTestInput(CURRENT_INPUT_GRID);
    $('#current_test_input_id_display').html(CURRENT_TEST_PAIR_INDEX + 1);
    $('#total_test_input_count_display').html(test.length);
}

function submitSolution() {
    syncFromEditionGridToDataGrid();
    reference_output = TEST_PAIRS[CURRENT_TEST_PAIR_INDEX]['output'];
    submitted_output = CURRENT_OUTPUT_GRID.grid;
    if (reference_output.length != submitted_output.length) {
        errorMsg('Wrong solution.');
        return
    }
    for (var i = 0; i < reference_output.length; i++){
        ref_row = reference_output[i];
        for (var j = 0; j < ref_row.length; j++){
            if (ref_row[j] != submitted_output[i][j]) {
                errorMsg('Wrong solution.');
                return
            }
        }

    }
    infoMsg('Correct solution!');
}

function fillTestInput(inputGrid) {
    jqInputGrid = $('#evaluation_input');
    fillJqGridWithData(jqInputGrid, inputGrid);
    fitCellsToContainer(jqInputGrid, inputGrid.height, inputGrid.width, 400, 400);
}

function copyToOutput() {
    syncFromEditionGridToDataGrid();
    CURRENT_OUTPUT_GRID = convertSerializedGridToGridObject(CURRENT_INPUT_GRID.grid);
    syncFromDataGridToEditionGrid();
    $('#output_grid_size').val(CURRENT_OUTPUT_GRID.height + 'x' + CURRENT_OUTPUT_GRID.width);
}

function initializeSelectable() {
    try {
        $('.selectable_grid').selectable('destroy');
    }
    catch (e) {
    }
    toolMode = $('input[name=tool_switching]:checked').val();
    if (toolMode == 'select') {
        infoMsg('Select some cells and click on a color to fill in, or press C to copy');
        $('.selectable_grid').selectable(
            {
                autoRefresh: false,
                filter: '> .row > .cell',
                start: function(event, ui) {
                    $('.ui-selected').each(function(i, e) {
                        $(e).removeClass('ui-selected');
                    });
                }
            }
        );
    }
}

// Initial event binding.

$(document).ready(function () {
    $('#symbol_picker').find('.symbol_preview').click(function(event) {
        symbol_preview = $(event.target);
        $('#symbol_picker').find('.symbol_preview').each(function(i, preview) {
            $(preview).removeClass('selected-symbol-preview');
        })
        symbol_preview.addClass('selected-symbol-preview');

        toolMode = $('input[name=tool_switching]:checked').val();
        if (toolMode == 'select') {
            $('.edition_grid').find('.ui-selected').each(function(i, cell) {
                symbol = getSelectedSymbol();
                setCellSymbol($(cell), symbol);
            });
        }
    });

    $('.edition_grid').each(function(i, jqGrid) {
        setUpEditionGridListeners($(jqGrid));
    });

    $('.load_task').on('change', function(event) {
        loadTaskFromFile(event);
    });

    $('.load_task').on('click', function(event) {
      event.target.value = "";
    });

    $('input[type=radio][name=tool_switching]').change(function() {
        initializeSelectable();
    });

    $('body').keydown(function(event) {
        // Copy and paste functionality.
        if (event.which == 67) {
            // Press C

            selected = $('.ui-selected');
            if (selected.length == 0) {
                return;
            }

            COPY_PASTE_DATA = [];
            for (var i = 0; i < selected.length; i ++) {
                x = parseInt($(selected[i]).attr('x'));
                y = parseInt($(selected[i]).attr('y'));
                symbol = parseInt($(selected[i]).attr('symbol'));
                COPY_PASTE_DATA.push([x, y, symbol]);
            }
            infoMsg('Cells copied! Select a target cell and press V to paste at location.');

        }
        if (event.which == 86) {
            // Press P
            if (COPY_PASTE_DATA.length == 0) {
                errorMsg('No data to paste.');
                return;
            }
            selected = $('.edition_grid').find('.ui-selected');
            if (selected.length == 0) {
                errorMsg('Select a target cell on the output grid.');
                return;
            }

            jqGrid = $(selected.parent().parent()[0]);

            if (selected.length == 1) {
                targetx = parseInt(selected.attr('x'));
                targety = parseInt(selected.attr('y'));

                xs = new Array();
                ys = new Array();
                symbols = new Array();

                for (var i = 0; i < COPY_PASTE_DATA.length; i ++) {
                    xs.push(COPY_PASTE_DATA[i][0]);
                    ys.push(COPY_PASTE_DATA[i][1]);
                    symbols.push(COPY_PASTE_DATA[i][2]);
                }

                minx = Math.min(...xs);
                miny = Math.min(...ys);
                for (var i = 0; i < xs.length; i ++) {
                    x = xs[i];
                    y = ys[i];
                    symbol = symbols[i];
                    newx = x - minx + targetx;
                    newy = y - miny + targety;
                    res = jqGrid.find('[x="' + newx + '"][y="' + newy + '"] ');
                    if (res.length == 1) {
                        cell = $(res[0]);
                        setCellSymbol(cell, symbol);
                    }
                }
            } else {
                errorMsg('Can only paste at a specific location; only select *one* cell as paste destination.');
            }
        }
    });
});
