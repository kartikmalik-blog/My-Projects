'''🐍 Phase 1: Python'''

'''1. Basic Python Challenge: The Data Validator
Goal: Write a function to check if a customer ID list is valid.

Skill Focus: Loops (for/while), conditional logic (if/else), and basic data structures (lists, strings).
'''

def validate_ids(id_lists):
    """Checks if customer IDs are valid (must be 8 character long and contain only digits)."""
    invalid_ids = []

    for customer_id in id_lists:
        #Check 1: Length must be 8
        if len(customer_id) != 8:
            invalid_ids.append(f"{customer_id} (Length Error)")
            continue

        #Check 2: Must be entirely digits
        if not customer_id.isdigit():
            invalid_ids.append(f"{customer_id} (Format Error)")
            continue

    if invalid_ids :
        print(f"Validation failed for {len(invalid_ids)} IDs.")
        return invalid_ids
    else:
        return "All IDs are valid !"
    
    # Test Data
customer_ids = ["12345678", "9876543A", "555", "11223344"]

# Run the validation
errors = validate_ids(customer_ids)
print(errors)