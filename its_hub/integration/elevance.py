import re
from typing import Tuple

def _elevance_projection_func(response: str) -> Tuple[str]:
    """Elevance projection function that extracts all meaningful nodes from the isr representation. 

    Input:
    task #1 {\n    %1 = members.ref_member\n    %2 = priorAuthorization.filter patient_member_id:%1 qualifier_phrase:\"HEIGHTS SURGERY CENTER\"\n    %3 = priorAuthorization.ref_authorizations filter:%2\n    %4 = priorAuthorization.show_authorizations priorAuthorizationResponse:%3\n    %5 = builtin.abstract_question dependency:%3 question:\"Was my procedure pre-approved?\"\n    %6 = builtin.show data:%5\n}

    Output:
    ['members.ref_member', 'priorAuthorization.filter patient_member_id:%1 qualifier_phrase:"HEIGHTS SURGERY CENTER"', 'priorAuthorization.ref_authorizations filter:%2', 'priorAuthorization.show_authorizations priorAuthorizationResponse:%3', 'builtin.abstract_question dependency:%3 question:"Was my procedure pre-approved?"', 'builtin.show data:%5']
    """
    nodes = []
    
    # Remove leading/trailing whitespace and normalize newlines
    response = response.strip()
    
    # Split into lines and process each line
    lines = response.split('\n')
    
    for line in lines:
        # Strip whitespace from each line
        line = line.strip()
        
        # Skip empty lines, task declarations, and braces
        if not line or line.startswith('task #') or line in ['{', '}']:
            continue
            
        # Match assignment patterns: %N = operation args...
        match = re.match(r'%\d+\s*=\s*(.+)$', line)
        if match:
            # Extract the operation part (everything after the = sign)
            operation = match.group(1).strip()
            nodes.append(operation)
    
    return tuple(nodes)
