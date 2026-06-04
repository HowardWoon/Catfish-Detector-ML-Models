import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import codecs

def execute_notebook():
    nb_path = 'WIA1006_OCC3_Catfish_Group7_Ultimate.ipynb'
    with codecs.open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    # We will execute all cells except the last one (Cell 25) which runs the infinite Flask server
    # Find the cell index that contains PyNgrok / Cell 25
    flask_cell_idx = -1
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'import pyngrok' in cell.source:
            flask_cell_idx = i
            break
            
    if flask_cell_idx != -1:
        # Clear the output of the flask cell just in case
        nb.cells[flask_cell_idx].outputs = []
        cells_to_run = nb.cells[:flask_cell_idx]
    else:
        cells_to_run = nb.cells

    print(f"Executing {len(cells_to_run)} cells...")
    
    # Create a temporary notebook with just the cells to run
    temp_nb = nbformat.v4.new_notebook()
    temp_nb.cells = cells_to_run
    
    ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
    
    try:
        ep.preprocess(temp_nb, {'metadata': {'path': './'}})
    except Exception as e:
        print(f"Error executing notebook: {str(e).encode('ascii', 'ignore').decode('ascii')}")
        # Even if it errors, we want to save the partial progress to see where it failed
        
    # Copy the outputs back to the original notebook
    for i in range(len(cells_to_run)):
        if nb.cells[i].cell_type == 'code':
            nb.cells[i].outputs = temp_nb.cells[i].outputs
            if hasattr(temp_nb.cells[i], 'execution_count'):
                nb.cells[i].execution_count = temp_nb.cells[i].execution_count
            
    with codecs.open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print("Notebook Execution and Saving Complete!")

if __name__ == '__main__':
    execute_notebook()
