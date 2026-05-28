import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import defaultdict

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def normalize_drug_name(name):
    """Normalize drug name into set of words"""
    # Remove special characters and convert to lowercase
    cleaned = name.lower()
    for char in ['(', ')', '/', '-']:
        cleaned = cleaned.replace(char, ' ')
    # Split into words and remove empty strings
    words = [word.strip() for word in cleaned.split() if word.strip()]
    return set(words)

def drugs_match(drug1_name, drug2_name):
    """Check if two drug names match based on word overlap"""
    words1 = normalize_drug_name(drug1_name)
    words2 = normalize_drug_name(drug2_name)
    # If any word appears in both sets, it's a match
    return len(words1.intersection(words2)) > 0

def load_extracted_data(extracted_json):
    """Parse LLM-extracted data"""
    data = {}
    uncertainty_data = {}
    
    for patient in extracted_json:
        code = patient['patient_code']
        data[code] = {}
        uncertainty_data[code] = {}
        
        for drug in patient['antimicrobial_usage']:
            drug_name = drug['antimicrobial_administered']
            data[code][drug_name] = safe_float(drug['total_mg_administered'])
            try:
                uncertainty_data[code][drug_name] = safe_float(drug['uncertainty_cv_percent'])
            except:
                print(f"no uncertainty for patient {code} drug {drug_name}, skipping")
                continue
    
    return data, uncertainty_data

def load_gold_standard(gold_json):
    """Parse gold standard data"""
    data = {}
    for patient in gold_json:
        code = patient['code']
        data[code] = {}
        i = 1
        while f'ABdrug{i}' in patient:
            drug_name = patient[f'ABdrug{i}']
            dose = safe_float(patient[f'totalmgadministered{i}'])
            data[code][drug_name] = dose
            i += 1
    return data

def match_data(extracted, gold_standard):
    """Match extracted data with gold standard using fuzzy matching"""
    y_true = []
    y_pred = []
    matches = []
    
    for code in gold_standard:
        if code not in extracted:
            continue
            
        for gold_drug_name, gold_dose in gold_standard[code].items():
            # Try to find matching drug in extracted data
            matched = False
            for extracted_drug_name, pred_dose in extracted[code].items():
                if gold_dose is None or pred_dose is None:
                    print("One or both of the doses are none, skipping")
                    continue
                if drugs_match(gold_drug_name, extracted_drug_name):
                    y_true.append(gold_dose)
                    y_pred.append(pred_dose)
                    error = abs(pred_dose - gold_dose)
                    pct_error = abs((pred_dose - gold_dose) / gold_dose * 100) if gold_dose != 0 else 0
                    matches.append({
                        'code': code,
                        'gold_drug': gold_drug_name,
                        'extracted_drug': extracted_drug_name,
                        'gold': gold_dose,
                        'predicted': pred_dose,
                        'error': error,
                        'percent_error': pct_error
                    })
                    matched = True
                    break  # Found a match, move to next gold drug
            
            if not matched:
                print(f"No match found: Patient {code}, Gold drug: {gold_drug_name}")
    
    return np.array(y_true), np.array(y_pred), matches

def calculate_uncertainty_metrics(uncertainty_data):
    """Calculate average uncertainty per patient and per drug"""
    
    # Calculate average uncertainty per patient
    patient_uncertainties = []
    for code, drugs in uncertainty_data.items():
        cv_values = [cv for cv in drugs.values() if cv is not None]
        if cv_values:
            patient_mean_cv = np.mean(cv_values)
            patient_uncertainties.append({
                'code': code,
                'mean_cv': patient_mean_cv,
                'num_drugs': len(cv_values)
            })
    
    # Calculate average uncertainty per drug WITH FUZZY MATCHING
    drug_uncertainties = defaultdict(list)
    drug_name_mapping = {}  # Keep track of which normalized name we're using
    
    for code, drugs in uncertainty_data.items():
        for drug_name, cv in drugs.items():
            if cv is not None:
                # Check if this drug matches any existing drug
                matched = False
                for existing_drug in drug_name_mapping.keys():
                    if drugs_match(drug_name, existing_drug):
                        # Use the existing normalized name
                        normalized_name = drug_name_mapping[existing_drug]
                        drug_uncertainties[normalized_name].append(cv)
                        matched = True
                        break
                
                if not matched:
                    # This is a new drug, use its name as the normalized version
                    drug_name_mapping[drug_name] = drug_name
                    drug_uncertainties[drug_name].append(cv)
    
    drug_uncertainty_summary = []
    for drug_name, cv_values in drug_uncertainties.items():
        drug_uncertainty_summary.append({
            'drug': drug_name,
            'mean_cv': np.mean(cv_values),
            'std_cv': np.std(cv_values),
            'min_cv': np.min(cv_values),
            'max_cv': np.max(cv_values),
            'count': len(cv_values)
        })
    
    return patient_uncertainties, drug_uncertainty_summary

def calculate_error_bins(matches):
    """Calculate number of predictions in different error bins"""
    error_bins = {
        'exact': 0,      # error = 0
        '<100': 0,       # 0 < error < 100
        '<500': 0,       # 100 <= error < 500
        '<1000': 0,      # 500 <= error < 1000
        '<5000': 0,      # 1000 <= error < 5000
        '<10000': 0,     # 5000 <= error < 10000
        '<25000': 0,     # 10000 <= error < 25000
        '>=25000': 0     # error >= 25000
    }
    
    for match in matches:
        error = match['error']
        if error == 0:
            error_bins['exact'] += 1
        elif error < 100:
            error_bins['<100'] += 1
        elif error < 500:
            error_bins['<500'] += 1
        elif error < 1000:
            error_bins['<1000'] += 1
        elif error < 5000:
            error_bins['<5000'] += 1
        elif error < 10000:
            error_bins['<10000'] += 1
        elif error < 25000:
            error_bins['<25000'] += 1
        else:
            error_bins['>=25000'] += 1
    
    return error_bins

def calculate_uncertainty_distribution(matches, uncertainty_data):
    """Calculate distribution of uncertainty values for matched comparisons"""
    uncertainty_bins = {
        '0': 0,           # uncertainty = 0
        '<5': 0,          # 0 < uncertainty < 5
        '<10': 0,         # 5 <= uncertainty < 10
        '<25': 0,         # 10 <= uncertainty < 25
        '<50': 0,         # 25 <= uncertainty < 50
        '<75': 0,         # 50 <= uncertainty < 75
        '>=75': 0         # uncertainty >= 75
    }
    
    uncertainties = []
    
    # Collect uncertainty values for matched comparisons
    for match in matches:
        code = match['code']
        extracted_drug = match['extracted_drug']
        
        # Find the uncertainty value for this match
        if code in uncertainty_data:
            # Find matching drug name (using fuzzy matching)
            uncertainty_value = None
            for drug_name, cv in uncertainty_data[code].items():
                if drugs_match(extracted_drug, drug_name) and cv is not None:
                    uncertainty_value = cv
                    break
            
            if uncertainty_value is not None:
                uncertainties.append(uncertainty_value)
                
                # Bin the uncertainty value
                if uncertainty_value == 0:
                    uncertainty_bins['0'] += 1
                elif uncertainty_value < 5:
                    uncertainty_bins['<5'] += 1
                elif uncertainty_value < 10:
                    uncertainty_bins['<10'] += 1
                elif uncertainty_value < 25:
                    uncertainty_bins['<25'] += 1
                elif uncertainty_value < 50:
                    uncertainty_bins['<50'] += 1
                elif uncertainty_value < 75:
                    uncertainty_bins['<75'] += 1
                else:
                    uncertainty_bins['>=75'] += 1
    
    return uncertainty_bins, uncertainties

# Main execution
if __name__ == "__main__":
    # Load your JSON files
    with open('./data/extracted_data.json', 'r') as f:
        extracted_json = json.load(f)
    
    with open('./data/mgadministered_gold_data.json', 'r') as f:
        gold_json = json.load(f)
    
    # Process data
    extracted, uncertainty_data = load_extracted_data(extracted_json)
    gold_standard = load_gold_standard(gold_json)
    y_true, y_pred, matches = match_data(extracted, gold_standard)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    median_error = np.median(np.abs(y_true - y_pred))
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Calculate uncertainty metrics
    patient_uncertainties, drug_uncertainty_summary = calculate_uncertainty_metrics(uncertainty_data)
    
    # Calculate error bins
    error_bins = calculate_error_bins(matches)

    # Calculate uncertainty distribution
    uncertainty_bins, uncertainties = calculate_uncertainty_distribution(matches, uncertainty_data)
    
    # Print results
    print("=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"Number of comparisons: {len(y_true)}")
    print(f"Mean Absolute Error: {mae:.2f} mg")
    print(f"Median Absolute Error: {median_error:.2f} mg")
    print(f"RMSE: {rmse:.2f} mg")
    print(f"MAPE: {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    print(f"Max Error: {max_error:.2f} mg")
    
    # Print error distribution
    print("\n" + "=" * 60)
    print("ERROR DISTRIBUTION")
    print("=" * 60)
    total = len(matches)
    print(f"Exact match (error = 0):           {error_bins['exact']:>5} ({error_bins['exact']/total*100:>5.1f}%)")
    print(f"Error < 100 mg:                    {error_bins['<100']:>5} ({error_bins['<100']/total*100:>5.1f}%)")
    print(f"Error 100-500 mg:                  {error_bins['<500']:>5} ({error_bins['<500']/total*100:>5.1f}%)")
    print(f"Error 500-1000 mg:                 {error_bins['<1000']:>5} ({error_bins['<1000']/total*100:>5.1f}%)")
    print(f"Error 1000-5000 mg:                {error_bins['<5000']:>5} ({error_bins['<5000']/total*100:>5.1f}%)")
    print(f"Error 5000-10000 mg:               {error_bins['<10000']:>5} ({error_bins['<10000']/total*100:>5.1f}%)")
    print(f"Error 10000-25000 mg:              {error_bins['<25000']:>5} ({error_bins['<25000']/total*100:>5.1f}%)")
    print(f"Error >= 25000 mg:                 {error_bins['>=25000']:>5} ({error_bins['>=25000']/total*100:>5.1f}%)")
    
    # Print uncertainty metrics
    print("\n" + "=" * 60)
    print("UNCERTAINTY METRICS")
    print("=" * 60)
    
    # Average uncertainty per patient
    if patient_uncertainties:
        overall_patient_uncertainty = np.mean([p['mean_cv'] for p in patient_uncertainties])
        print(f"\nAverage Uncertainty Per Patient (mean of patient means):")
        print(f"  Overall: {overall_patient_uncertainty:.2f}% CV")
        print(f"  Std Dev: {np.std([p['mean_cv'] for p in patient_uncertainties]):.2f}% CV")
        print(f"  Min: {np.min([p['mean_cv'] for p in patient_uncertainties]):.2f}% CV")
        print(f"  Max: {np.max([p['mean_cv'] for p in patient_uncertainties]):.2f}% CV")
        print(f"  Number of patients: {len(patient_uncertainties)}")
        
        # Show top 5 patients with highest uncertainty
        print("\n  Top 5 Patients with Highest Uncertainty:")
        sorted_patients = sorted(patient_uncertainties, key=lambda x: x['mean_cv'], reverse=True)
        for i, p in enumerate(sorted_patients[:5], 1):
            print(f"    {i}. Patient {p['code']}: {p['mean_cv']:.2f}% CV ({p['num_drugs']} drugs)")
    
    # Average uncertainty per drug
    if drug_uncertainty_summary:
        drug_uncertainty_summary = sorted(drug_uncertainty_summary, key=lambda x: x['mean_cv'], reverse=True)
        print(f"\nAverage Uncertainty Per Drug:")
        print(f"  {'Drug':<30} {'Mean CV':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Count':<10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for drug in drug_uncertainty_summary:
            print(f"  {drug['drug']:<30} {drug['mean_cv']:>8.2f}% {drug['std_cv']:>8.2f}% {drug['min_cv']:>8.2f}% {drug['max_cv']:>8.2f}% {drug['count']:>10}")
        
        overall_drug_uncertainty = np.mean([d['mean_cv'] for d in drug_uncertainty_summary])
        print(f"\n  Overall mean across all drugs: {overall_drug_uncertainty:.2f}% CV")
    
        # Print uncertainty distribution
    print("\n" + "=" * 60)
    print("UNCERTAINTY DISTRIBUTION")
    print("=" * 60)
    if uncertainties:
        total_with_uncertainty = len(uncertainties)
        print(f"Total comparisons with uncertainty data: {total_with_uncertainty}")
        print(f"\nUncertainty = 0%:                  {uncertainty_bins['0']:>5} ({uncertainty_bins['0']/total_with_uncertainty*100:>5.1f}%)")
        print(f"Uncertainty < 5%:                  {uncertainty_bins['<5']:>5} ({uncertainty_bins['<5']/total_with_uncertainty*100:>5.1f}%)")
        print(f"Uncertainty 5-10%:                 {uncertainty_bins['<10']:>5} ({uncertainty_bins['<10']/total_with_uncertainty*100:>5.1f}%)")
        print(f"Uncertainty 10-25%:                {uncertainty_bins['<25']:>5} ({uncertainty_bins['<25']/total_with_uncertainty*100:>5.1f}%)")
        print(f"Uncertainty 25-50%:                {uncertainty_bins['<50']:>5} ({uncertainty_bins['<50']/total_with_uncertainty*100:>5.1f}%)")
        print(f"Uncertainty 50-75%:                {uncertainty_bins['<75']:>5} ({uncertainty_bins['<75']/total_with_uncertainty*100:>5.1f}%)")
        print(f"Uncertainty >= 75%:                {uncertainty_bins['>=75']:>5} ({uncertainty_bins['>=75']/total_with_uncertainty*100:>5.1f}%)")
        
        print(f"\nUncertainty Statistics:")
        print(f"  Mean: {np.mean(uncertainties):.2f}% CV")
        print(f"  Median: {np.median(uncertainties):.2f}% CV")
        print(f"  Std Dev: {np.std(uncertainties):.2f}% CV")
        print(f"  Min: {np.min(uncertainties):.2f}% CV")
        print(f"  Max: {np.max(uncertainties):.2f}% CV")
    else:
        print("No uncertainty data available for matched comparisons")
    
    print("\n" + "=" * 60)
    print("TOP 10 LARGEST ERRORS")
    print("=" * 60)
    sorted_matches = sorted(matches, key=lambda x: x['error'], reverse=True)
    for i, m in enumerate(sorted_matches[:10], 1):
        print(f"{i}. Patient {m['code']}")
        print(f"   Gold: {m['gold_drug']} | Extracted: {m['extracted_drug']}")
        print(f"   Gold dose: {m['gold']:.2f} mg | Predicted: {m['predicted']:.2f} mg")
        print(f"   Error: {m['error']:.2f} mg ({m['percent_error']:.1f}%)\n")