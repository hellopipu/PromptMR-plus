function [unique_indices, duplicates_found] = findUniqueSlices(imageData, dim_to_check)
%findUniqueSlices Checks for and returns the indices of unique slices in a matrix.
%
%   [unique_indices, duplicates_found] = findUniqueSlices(imageData, dim_to_check)
%   analyzes the input matrix 'imageData' along the specified dimension
%   'dim_to_check' to identify and report duplicate slices.
%
%   Inputs:
%       imageData    - The N-dimensional matrix to check (e.g., 4D MRI data).
%       dim_to_check - The dimension along which to check for duplicates (e.g., 3).
%
%   Outputs:
%       unique_indices   - A row vector containing the indices of the first
%                          occurrence of each unique slice.
%       duplicates_found - A logical flag (true or false) indicating if any
%                          duplicates were found.
%
%   Example:
%       % Create sample data with duplicates
%       data = rand(10, 10, 5);
%       data(:,:,4) = data(:,:,1); % Make slice 4 a copy of slice 1
%
%       % Find the unique slices along the 3rd dimension
%       unique_indices = findUniqueSlices(data, 3);
%
%       % Create a new matrix with only the unique slices
%       cleaned_data = data(:,:,unique_indices);
%
%       disp(unique_indices);
%       % Expected output: [1 2 3 5]

% =========================================================================

    % --- 1. Input Validation ---
    if nargin < 2
        error('This function requires two arguments: imageData and dim_to_check.');
    end
    if dim_to_check > ndims(imageData) || dim_to_check < 1
        error('dim_to_check must be a valid dimension of the input data.');
    end

    fprintf('--- Starting Duplicate Check along Dimension %d ---\n', dim_to_check);
    
    % --- 2. Perform the Duplicate Check ---
    numSlices = size(imageData, dim_to_check);
    is_unique = true(1, numSlices); % Assume all slices are unique initially
    duplicates_found = false;

    % Loop through each slice
    for i = 1:(numSlices - 1)
        % If slice 'i' has already been found to be a duplicate, skip it
        if ~is_unique(i)
            continue;
        end
        
        % Extract slice 'i' once for the inner loop comparison
        indices_i = repmat({':'}, 1, ndims(imageData));
        indices_i{dim_to_check} = i;
        slice_i = imageData(indices_i{:});

        % In a nested loop, compare slice 'i' with all subsequent slices 'j'
        for j = (i + 1):numSlices
            % Extract slice 'j'
            indices_j = repmat({':'}, 1, ndims(imageData));
            indices_j{dim_to_check} = j;
            slice_j = imageData(indices_j{:});
            
            % If the slices are identical, mark slice 'j' as not unique
            if isequal(slice_i, slice_j)
                fprintf('  -> Found Duplicate: Slice %d is identical to Slice %d.\n', j, i);
                is_unique(j) = false;
                duplicates_found = true;
            end
        end
    end

    % The final list of unique indices are the ones still marked as true
    unique_indices = find(is_unique);
    
    % --- 3. Final Report ---
    fprintf('\n--- Check Complete ---\n');
    if duplicates_found
        fprintf('Duplicates were found. Returning indices of unique slices.\n');
    else
        fprintf('No duplicate slices were found along dimension %d.\n', dim_to_check);
    end
    fprintf('Unique slice indices: [%s]\n', num2str(unique_indices));
    
end
