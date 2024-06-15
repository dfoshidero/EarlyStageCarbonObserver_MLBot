import pandas as pd
import spacy
from spacy.matcher import Matcher

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Create a matcher for custom entities
matcher = Matcher(nlp.vocab)

# Function to create patterns from a list of terms
def create_patterns(terms):
    return [[{"LOWER": term}] for term in terms]

# Create patterns for all materials, building types, and element buildups
material_patterns = create_patterns(all_materials)
building_type_patterns = create_patterns(all_building_types)
element_buildup_patterns = create_patterns(all_element_buildups)

# Add patterns to the matcher
matcher.add("MATERIAL", material_patterns)
matcher.add("BUILDING_TYPE", building_type_patterns)
matcher.add("ELEMENT_BUILDUP", element_buildup_patterns)

# Parse input description
def parse_description(description):
    doc = nlp(description)
    size = 0
    size_unit = 'square meters'
    materials = []
    building_types = []
    element_buildups = []
    location = 'unknown'
    
    matches = matcher(doc)
    for match_id, start, end in matches:
        match_label = nlp.vocab.strings[match_id]
        if match_label == 'MATERIAL':
            materials.append(doc[start:end].text.lower())
        elif match_label == 'BUILDING_TYPE':
            building_types.append(doc[start:end].text.lower())
        elif match_label == 'ELEMENT_BUILDUP':
            element_buildups.append(doc[start:end].text.lower())
    
    for ent in doc.ents:
        if ent.label_ == 'QUANTITY':
            size = float(ent.text.replace(',', '').split()[0])
            size_unit = ' '.join(ent.text.split()[1:])
        elif ent.label_ == 'GPE':
            location = ent.text.lower()
        elif ent.label_ in ('ORG', 'FAC', 'PRODUCT'):
            token_text = ent.text.lower()
            if not building_types and any(nlp(token_text).similarity(nlp(b_type)) > 0.75 for b_type in all_building_types):
                building_types.append(token_text)
            if not materials and any(nlp(token_text).similarity(nlp(m)) > 0.75 for m in all_materials):
                materials.append(token_text)
    
    if 'square feet' in size_unit or 'sq ft' in size_unit:
        size *= 0.092903
    
    return {
        'size': size,
        'materials': materials,
        'building_types': building_types,
        'element_buildups': element_buildups,
        'location': location
}

# Example usage
if __name__ == "__main__":
    description = "I am constructing a school building in New York that is 2000 square metres, primarily made of timber and steel. It will have pile foundations"
    parsed_info = parse_description(description)
    print(parsed_info)
