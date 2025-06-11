# ==============================================================================
# 1. IMPORTS AND DEPENDENCIES
# ==============================================================================
# Standard library imports
import os
import pickle
import zipfile
import io
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

# Third-party library imports
import streamlit as st
import pandas as pd
import numpy as np
import torch
import gpytorch
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.gridspec import GridSpec
from PIL import Image

# Chemistry-specific imports (requires rdkit-pypi)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, Descriptors, rdDepictor, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import rdRGroupDecomposition, rdMMPA
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Local application-specific imports

# ==============================================================================
# 2. APPLICATION CONFIGURATION
# ==============================================================================
PAGE_TITLE = "SHAP-Guided Drug Discovery"
PAGE_ICON = "🧬"
FINGERPRINT_RADIUS = 4
FINGERPRINT_BITS = 4096
AURA_HIGHLIGHT_COLOR = (1.0, 0.4, 0.8, 0.25)
AURA_HIGHLIGHT_RADIUS = 0.65

CUSTOM_APP_CSS = """
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .sub-header {font-size: 1.5rem; color: #ff7f0e; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #ff7f0e;}
    .info-box {background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;}
    .stPlotlyChart {min-height: 550px;}
</style>
"""

# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================
def create_molecule_aura_image(molecule, highlight_atoms, highlight_bonds):
    if not molecule: return None
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
    drawer.drawOptions().padding = 0.1
    atom_highlight_map = {idx: [AURA_HIGHLIGHT_COLOR] for idx in highlight_atoms}
    bond_highlight_map = {idx: [AURA_HIGHLIGHT_COLOR] for idx in highlight_bonds}
    atom_radii_map = {idx: AURA_HIGHLIGHT_RADIUS for idx in highlight_atoms}
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=atom_highlight_map, highlightBonds=bond_highlight_map, highlightAtomRadii=atom_radii_map)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))

# ==============================================================================
# 4. CORE ANALYSIS & GENERATOR CLASSES
# ==============================================================================
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def smiles_to_ecfp8(df, smiles_col):
    """Simple ECFP fingerprint generation"""
    from rdkit.Chem import AllChem
    fingerprints = []
    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=4096)
            fingerprints.append(fp)
        else:
            fingerprints.append(None)
    return np.array(fingerprints)


class ChemicalFragmentMapper:
    def __init__(self, analysis_results, dataset_df, fingerprint_func):
        self.analysis_results, self.dataset_df, self.fingerprint_func = analysis_results, dataset_df, fingerprint_func
        self.fp_radius, self.fp_bits = FINGERPRINT_RADIUS, FINGERPRINT_BITS
        self.fingerprints = self.fingerprint_func(self.dataset_df, 'SMILES')

    def extract_fragments_for_features(self, target, protocol, top_n, max_mols):
        df = self.analysis_results['feature_evolution']
        target_df = self.dataset_df[self.dataset_df['Target'].str.strip().str.upper() == target.upper()].copy().reset_index(drop=True)
        imp = df.groupby('feature_index')['importance'].mean()
        top_feats = imp.nlargest(top_n)
        frag_data = {}
        for rank, (idx, avg_imp) in enumerate(top_feats.items(), 1):
            mols_with_feat = [
                {'mol_idx': i, 'smiles': r['SMILES'], 'affinity': r['affinity']}
                for i, r in target_df.iterrows()
                if i < self.fingerprints.shape[0] and idx < self.fingerprints.shape[1] and self.fingerprints[i, idx] == 1
            ]
            if not mols_with_feat: continue
            mols_with_feat.sort(key=lambda x: x['affinity'], reverse=True)
            feat_frags = self._extract_substructures(idx, mols_with_feat[:max_mols])
            frag_data[idx] = {'feature_index': idx, 'average_importance': avg_imp, 'rank': rank, 'molecules_with_feature': mols_with_feat, 'fragments': feat_frags}
        return frag_data

    def _extract_smiles_from_highlighted_atoms(self, mol, highlighted_atoms):
        """
        ROBUST SMILES extraction with multiple fallback methods.
        This is the FIXED version that actually works!
        """
        if not highlighted_atoms:
            return None
        
        print(f"    Trying to extract SMILES from {len(highlighted_atoms)} highlighted atoms...")
        
        # Method 1: Connected subgraph approach
        try:
            print("      Method 1: Connected subgraph...")
            bonds_to_include = []
            for bond in mol.GetBonds():
                if (bond.GetBeginAtomIdx() in highlighted_atoms and 
                    bond.GetEndAtomIdx() in highlighted_atoms):
                    bonds_to_include.append(bond.GetIdx())
            
            if bonds_to_include:
                submol = Chem.PathToSubmol(mol, bonds_to_include)
                if submol and submol.GetNumAtoms() > 0:
                    try:
                        Chem.SanitizeMol(submol)
                        fragment_smiles = Chem.MolToSmiles(submol, canonical=True)
                        if fragment_smiles and len(fragment_smiles) > 1:
                            print(f"      ✅ Method 1 SUCCESS: {fragment_smiles}")
                            return fragment_smiles
                    except:
                        pass
            print("      ❌ Method 1 failed")
        except Exception as e:
            print(f"      ❌ Method 1 failed: {e}")
        
        # Method 2: RWMol construction
        try:
            print("      Method 2: RWMol construction...")
            
            # Create new molecule with highlighted atoms
            fragment_mol = Chem.RWMol()
            old_to_new_idx = {}
            
            # Add atoms
            for i, atom_idx in enumerate(highlighted_atoms):
                atom = mol.GetAtomWithIdx(atom_idx)
                new_atom = Chem.Atom(atom.GetAtomicNum())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                if atom.GetIsAromatic():
                    new_atom.SetIsAromatic(True)
                new_idx = fragment_mol.AddAtom(new_atom)
                old_to_new_idx[atom_idx] = new_idx
            
            # Add bonds between highlighted atoms
            bonds_added = 0
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                if begin_idx in old_to_new_idx and end_idx in old_to_new_idx:
                    try:
                        fragment_mol.AddBond(
                            old_to_new_idx[begin_idx], 
                            old_to_new_idx[end_idx], 
                            bond.GetBondType()
                        )
                        bonds_added += 1
                    except:
                        pass
            
            if bonds_added > 0:
                try:
                    final_mol = fragment_mol.GetMol()
                    if final_mol:
                        Chem.SanitizeMol(final_mol)
                        fragment_smiles = Chem.MolToSmiles(final_mol, canonical=True)
                        if fragment_smiles and len(fragment_smiles) > 1:
                            print(f"      ✅ Method 2 SUCCESS: {fragment_smiles}")
                            return fragment_smiles
                except:
                    pass
            print("      ❌ Method 2 failed")
        except Exception as e:
            print(f"      ❌ Method 2 failed: {e}")
        
        # Method 3: Empirical formula approach
        try:
            print("      Method 3: Empirical formula...")
            
            element_counts = {}
            for atom_idx in highlighted_atoms:
                atom = mol.GetAtomWithIdx(atom_idx)
                symbol = atom.GetSymbol()
                element_counts[symbol] = element_counts.get(symbol, 0) + 1
            
            # Create molecular formula
            formula_parts = []
            for element in ['C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P']:
                if element in element_counts:
                    count = element_counts[element]
                    if count == 1:
                        formula_parts.append(element)
                    else:
                        formula_parts.append(f"{element}{count}")
            
            # Add any remaining elements
            for element, count in element_counts.items():
                if element not in ['C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P']:
                    if count == 1:
                        formula_parts.append(element)
                    else:
                        formula_parts.append(f"{element}{count}")
            
            if formula_parts:
                formula = "".join(formula_parts)
                print(f"      ✅ Method 3 SUCCESS: {formula}")
                return formula
            
            print("      ❌ Method 3 failed")
        except Exception as e:
            print(f"      ❌ Method 3 failed: {e}")
        
        # Method 4: Last resort - atom count description
        try:
            print("      Method 4: Last resort...")
            atom_count = len(highlighted_atoms)
            fragment_desc = f"Fragment_{atom_count}_atoms"
            print(f"      ✅ Method 4 SUCCESS: {fragment_desc}")
            return fragment_desc
        except:
            pass
        
        print("      ❌ ALL METHODS FAILED")
        return None

    # ENHANCED VERSION with Streamlit progress indicators
# Replace your _extract_substructures method with this version that shows progress

    def _extract_substructures(self, feat_idx, mols):
        """Fixed substructure extraction with visible progress."""
        fragments_info = {'common_smiles': [], 'molecule_highlights': []}
        
        # Create progress container in Streamlit
        progress_container = st.empty()
        status_container = st.empty()
        
        progress_container.info(f"🔍 Processing feature {feat_idx} with {len(mols)} molecules...")
        
        successful_extractions = 0
        highlight_only_count = 0
        
        for i, mol_info in enumerate(mols):
            # Update progress
            progress = (i + 1) / len(mols)
            status_container.progress(progress, text=f"Analyzing molecule {i+1}/{len(mols)} (Affinity: {mol_info['affinity']:.2f})")
            
            mol = Chem.MolFromSmiles(mol_info['smiles'])
            if not mol: continue
            rdDepictor.Compute2DCoords(mol)
            
            bit_info = {}
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.fp_radius, nBits=self.fp_bits, bitInfo=bit_info)
            
            if feat_idx in bit_info:
                fragment_found_for_mol = False
                for atom_idx, radius in bit_info[feat_idx]:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                    if not env: continue
                    
                    bonds = list(env)
                    atoms = list(set([atom_idx] + [mol.GetBondWithIdx(b).GetBeginAtomIdx() for b in bonds] + [mol.GetBondWithIdx(b).GetEndAtomIdx() for b in bonds]))
                    
                    # Use the ROBUST method to extract SMILES
                    fragment_smiles = self._extract_smiles_from_highlighted_atoms(mol, atoms)
                    
                    if fragment_smiles:
                        fragments_info['common_smiles'].append(fragment_smiles)
                        fragments_info['molecule_highlights'].append({
                            'mol': mol, 
                            'highlight_atoms': atoms, 
                            'highlight_bonds': bonds, 
                            'parent_info': mol_info, 
                            'fragment_smiles': fragment_smiles
                        })
                        fragment_found_for_mol = True
                        successful_extractions += 1
                        break # Success for this molecule
                
                if not fragment_found_for_mol: # Fallback if no valid SMILES was generated
                    highlight_only_count += 1
                    try:
                        atom_idx, radius = bit_info[feat_idx][0]
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx) or []
                        bonds = list(env)
                        atoms = list(set([atom_idx] + [mol.GetBondWithIdx(b).GetBeginAtomIdx() for b in bonds] + [mol.GetBondWithIdx(b).GetEndAtomIdx() for b in bonds]))
                        fragments_info['molecule_highlights'].append({
                            'mol': mol, 
                            'highlight_atoms': atoms, 
                            'highlight_bonds': bonds, 
                            'parent_info': mol_info, 
                            'fragment_smiles': "Highlight_Only"
                        })
                    except:
                        pass
        
        # Clear progress indicators
        progress_container.empty()
        status_container.empty()
        
        # Generate most_common list and show results
        if fragments_info['common_smiles']: 
            fragments_info['most_common'] = Counter(fragments_info['common_smiles']).most_common(5)
            st.success(f"✅ **Feature {feat_idx}**: Extracted {successful_extractions} fragment SMILES, {highlight_only_count} highlights only")
            
            # Show the most common fragments found
            with st.expander("🧪 Fragment Extraction Results", expanded=False):
                for i, (frag_smiles, count) in enumerate(fragments_info['most_common'][:3]):
                    st.write(f"**{i+1}.** `{frag_smiles}` (found {count} times)")
        elif fragments_info['molecule_highlights']: 
            fragments_info['most_common'] = [("No SMILES Extracted", len(fragments_info['molecule_highlights']))]
            st.warning(f"⚠️ **Feature {feat_idx}**: No fragment SMILES extracted, but created {len(fragments_info['molecule_highlights'])} highlights")
        
        return fragments_info

    # SIMPLIFIED VERSION of _extract_smiles_from_highlighted_atoms for better debugging
    def _extract_smiles_from_highlighted_atoms(self, mol, highlighted_atoms):
        """
        STREAMLINED SMILES extraction - focuses on the most reliable methods.
        """
        if not highlighted_atoms:
            return None
        
        # Method 1: Connected subgraph approach (most reliable)
        try:
            bonds_to_include = []
            for bond in mol.GetBonds():
                if (bond.GetBeginAtomIdx() in highlighted_atoms and 
                    bond.GetEndAtomIdx() in highlighted_atoms):
                    bonds_to_include.append(bond.GetIdx())
            
            if bonds_to_include:
                submol = Chem.PathToSubmol(mol, bonds_to_include)
                if submol and submol.GetNumAtoms() > 0:
                    try:
                        Chem.SanitizeMol(submol)
                        fragment_smiles = Chem.MolToSmiles(submol, canonical=True)
                        if fragment_smiles and len(fragment_smiles) > 1:
                            return fragment_smiles
                    except:
                        pass
        except:
            pass
        
        # Method 2: Empirical formula approach (fallback)
        try:
            element_counts = {}
            for atom_idx in highlighted_atoms:
                atom = mol.GetAtomWithIdx(atom_idx)
                symbol = atom.GetSymbol()
                element_counts[symbol] = element_counts.get(symbol, 0) + 1
            
            # Create molecular formula
            formula_parts = []
            for element in ['C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P']:
                if element in element_counts:
                    count = element_counts[element]
                    if count == 1:
                        formula_parts.append(element)
                    else:
                        formula_parts.append(f"{element}{count}")
            
            if formula_parts:
                formula = "".join(formula_parts)
                return formula
        except:
            pass
        
        # Method 3: Last resort - descriptive text
        try:
            atom_count = len(highlighted_atoms)
            return f"Fragment_{atom_count}_atoms"
        except:
            pass
        
        return None

    def _create_fallback_gallery(self, frags_info):
        """Create fallback display when fragment extraction fails."""
        num_highlights = len(frags_info.get('molecule_highlights', []))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, f"Important Feature Detected\n\nFound in {num_highlights} molecules\n\n(Fragment structure could not\nbe automatically extracted)", 
               ha='center', va='center', fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"))
        ax.axis('off')
        return fig

    def _create_text_fragment_image(self, fragment_text):
        """
        Create a text-based image for fragments that can't be parsed as SMILES.
        """
        try:
            import matplotlib.pyplot as plt
            from io import BytesIO
            from PIL import Image
            
            fig, ax = plt.subplots(figsize=(3, 2))
            
            # Determine the type of fragment representation
            if fragment_text.startswith("Fragment_") and "atoms" in fragment_text:
                display_text = f"Chemical Fragment\n\n{fragment_text.replace('_', ' ')}"
                color = "lightblue"
            elif any(char.isdigit() for char in fragment_text) and len(fragment_text) < 20:
                display_text = f"Molecular Formula\n\n{fragment_text}"
                color = "lightgreen"
            else:
                display_text = f"Fragment Pattern\n\n{fragment_text}"
                color = "lightyellow"
            
            ax.text(0.5, 0.5, display_text, 
                ha='center', va='center', fontsize=10, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="gray"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title("Fragment Representation", fontsize=12, weight='bold')
            
            # Convert to PIL Image
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            
            return img
        except:
            return None

    def get_comprehensive_fragment_gallery(self, frags_info):
        """Enhanced gallery with better fallback handling."""
        if not frags_info.get('most_common') or frags_info['most_common'][0][0] in ["No SMILES Extracted", "Highlight_Only"]:
            return self._create_fallback_gallery(frags_info)
        
        mols, legends = [], []
        for frag_smiles, count in frags_info['most_common']:
            try:
                mol = Chem.MolFromSmiles(frag_smiles)
                if mol: 
                    mols.append(mol)
                    legends.append(f"Count: {count}\n{frag_smiles}")
            except:
                # Handle non-SMILES fragments
                pass
        
        if mols:
            return Draw.MolsToGridImage(mols, molsPerRow=min(3, len(mols)), subImgSize=(200, 200), legends=legends, useSVG=False)
        else:
            return self._create_fallback_gallery(frags_info)

    def get_integrated_fragment_display(self, frags_info, max_examples=3):
        """
        Enhanced integrated display that handles different types of fragment representations.
        """
        if not frags_info.get('most_common'):
            return None, None
        
        most_common_frag_smiles = frags_info['most_common'][0][0]
        
        # Handle special cases
        if most_common_frag_smiles in ["No SMILES Extracted", "No_Fragment_SMILES", "Highlight_Only"]:
            return None, None
        
        # Find matching highlights
        matching_highlights = [
            h for h in frags_info['molecule_highlights'] 
            if h.get('fragment_smiles') == most_common_frag_smiles
        ][:max_examples]
        
        # Try to create fragment molecule from SMILES
        isolated_img = None
        try:
            frag_mol = Chem.MolFromSmiles(most_common_frag_smiles)
            if frag_mol:
                isolated_img = Draw.MolToImage(frag_mol, size=(200, 200))
            else:
                # If SMILES parsing fails, create a text-based image
                isolated_img = self._create_text_fragment_image(most_common_frag_smiles)
        except:
            # Fallback to text-based representation
            isolated_img = self._create_text_fragment_image(most_common_frag_smiles)
        
        # Generate parent images
        parent_images = []
        for highlight_info in matching_highlights:
            parent_img = create_molecule_aura_image(
                highlight_info['mol'], 
                highlight_info['highlight_atoms'], 
                highlight_info['highlight_bonds']
            )
            if parent_img:
                parent_images.append({
                    'image': parent_img,
                    'affinity': highlight_info['parent_info']['affinity'],
                    'smiles': highlight_info['parent_info']['smiles']
                })
        
        return isolated_img, parent_images

    def get_affinity_plot_fig(self, feat_data):
        """Unchanged affinity plotting method."""
        affinities = [m['affinity'] for m in feat_data['molecules_with_feature']]
        if len(affinities) < 3: return None
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(affinities, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        mean, std = np.mean(affinities), np.std(affinities)
        ax.axvline(mean, color='red', ls='--', label=f'Mean: {mean:.2f} ± {std:.2f}')
        ax.set_title(f'Affinity (Feat. {feat_data["feature_index"]})')
        ax.set_xlabel('Binding Affinity')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        return fig

class MolecularDesignTemplateGenerator:
    def generate_scaffolds(self, target):
        if target == "TYK2": return {'quinazoline': {'smiles': 'c1cc(F)cc2c1ncnc2Cl', 'name': 'Fluorochloroquinazoline'}, 'pyrimidine': {'smiles': 'c1nc(N)nc(c(F)c(F)c1)c2ccccc2F', 'name': 'Trifluoropyrimidine'}}
        elif target == "USP7": return {'quinoxaline': {'smiles': 'O=C(N1CCN(CC1)c2cnc3ccccc3n2)c4cccnc4', 'name': 'Carbonyl quinoxaline'}, 'pyrazine': {'smiles': 'CCN(CC)c1cnc(nc1)C(=O)Nc2ccc3ccccc3c2', 'name': 'Pyrazine diethylamine'}}
        return {}
    def calculate_properties(self, smiles):
        mol = Chem.MolFromSmiles(smiles);
        if not mol: return {}
        return {'MW': Descriptors.MolWt(mol), 'LogP': Descriptors.MolLogP(mol), 'HBD': Descriptors.NumHDonors(mol), 'HBA': Descriptors.NumHAcceptors(mol), 'TPSA': Descriptors.TPSA(mol), 'RotBonds': Descriptors.NumRotatableBonds(mol)}
    def get_scaffold_grid_image(self, scaffolds):
        mols = [Chem.MolFromSmiles(data['smiles']) for data in scaffolds.values()]; legends = [data['name'] for data in scaffolds.values()]
        return Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(300, 300), legends=legends, useSVG=False) if mols else None

class PublicationFigureGenerator:
    def __init__(self, fragment_mappings, datasets):
        self.fragment_mappings, self.datasets = fragment_mappings, datasets
    def _get_representative_molecules(self, target, protocol, n_examples=3):
        mapper, fragments = self.fragment_mappings.get(f"{target}_{protocol}", (None, None))
        if not fragments: return []
        examples, used_molecules = [], set()
        sorted_features = sorted(fragments.items(), key=lambda item: item[1]['average_importance'], reverse=True)
        for feature_idx, feature_data in sorted_features:
            if len(examples) >= n_examples: break
            if 'molecule_highlights' in feature_data['fragments']:
                for highlight_info in feature_data['fragments']['molecule_highlights']:
                    if highlight_info['parent_info']['smiles'] not in used_molecules:
                        examples.append({**highlight_info, 'importance': feature_data['average_importance'], 'feature_idx': feature_idx})
                        used_molecules.add(highlight_info['parent_info']['smiles']); break
        return examples
    def create_main_figure(self, target, protocol):
        fig = plt.figure(figsize=(15, 5)); grid_spec = GridSpec(1, 3, figure=fig, wspace=0.1)
        examples = self._get_representative_molecules(target, protocol, 3)
        if not examples: return None
        for i, example in enumerate(examples):
            ax = fig.add_subplot(grid_spec[0, i])
            aura_image = create_molecule_aura_image(example['mol'], example['highlight_atoms'], example['highlight_bonds'])
            if aura_image: ax.imshow(aura_image)
            ax.axis('off'); ax.set_title(f"Feat. {example['feature_idx']} | SHAP: {example['importance']:.3f}\nAffinity: {example['parent_info']['affinity']:.2f}", fontsize=10)
        fig.suptitle(f"Key SHAP-Identified Features for {target} ({protocol})", fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95]); return fig

class SARAnalyzer:
    def __init__(self, dataset_df):
        self.df = dataset_df.copy()
        self.df['mol'] = self.df['SMILES'].apply(Chem.MolFromSmiles)
        self.df.dropna(subset=['mol'], inplace=True)
        self.df['mol_id'] = self.df.index.astype(str)
    def analyze_r_groups(self):
        scaffolds = [MurckoScaffold.GetScaffoldForMol(m) for m in self.df['mol']]
        most_common_scaffold_smarts = Counter([Chem.MolToSmarts(s) for s in scaffolds]).most_common(1)[0][0]
        core = Chem.MolFromSmarts(most_common_scaffold_smarts)
        decomp, unmatched = rdRGroupDecomposition.RGroupDecompose([core], list(self.df['mol']), asSmiles=True)
        r_group_df = pd.DataFrame(decomp).join(self.df.reset_index())
        return core, r_group_df
    # FIXED analyze_activity_cliffs method for the SARAnalyzer class
# Replace your existing analyze_activity_cliffs method with this corrected version

    def analyze_activity_cliffs(self, activity_threshold=1.0):
        """
        Fixed activity cliff analysis using correct rdMMPA.FragmentMol usage.
        """
        # Collect all fragments from all molecules
        all_fragments = []
        
        for idx, row in self.df.iterrows():
            mol_id = row['mol_id']
            mol = row['mol']
            
            try:
                # Fragment each molecule individually
                mol_fragments = rdMMPA.FragmentMol(
                    mol, 
                    minCuts=1, 
                    maxCuts=3, 
                    resultsAsMols=False
                )
                
                # Add molecule ID to each fragment
                for frag in mol_fragments:
                    if len(frag) >= 2:  # Should have core and side chains
                        core_smiles = frag[0]
                        side_chain_smiles = frag[1] if len(frag) > 1 else ""
                        all_fragments.append((mol_id, core_smiles, side_chain_smiles))
                        
            except Exception as e:
                # Skip molecules that can't be fragmented
                print(f"Could not fragment molecule {mol_id}: {e}")
                continue
        
        # Group fragments by core structure
        mmp_index = defaultdict(list)
        for mol_id, core_smiles, side_chain_smiles in all_fragments:
            mmp_index[core_smiles].append((mol_id, side_chain_smiles))
        
        # Find activity cliffs
        cliffs = []
        activity_map = pd.Series(self.df.affinity.values, index=self.df.mol_id).to_dict()
        mol_map = pd.Series(self.df.mol.values, index=self.df.mol_id).to_dict()
        
        for core_smiles, pairs in mmp_index.items():
            if len(pairs) > 1:
                # Compare all pairs sharing the same core
                for i in range(len(pairs)):
                    for j in range(i + 1, len(pairs)):
                        id1, r1 = pairs[i]
                        id2, r2 = pairs[j]
                        
                        act1 = activity_map.get(id1)
                        act2 = activity_map.get(id2)
                        
                        if act1 is not None and act2 is not None:
                            delta = abs(act1 - act2)
                            if delta >= activity_threshold:
                                cliffs.append({
                                    'mol1': mol_map[id1], 
                                    'mol2': mol_map[id2], 
                                    'act1': act1, 
                                    'act2': act2, 
                                    'delta': delta, 
                                    'transform': f"{r1} >> {r2}",
                                    'core': core_smiles
                                })
        
        # Return top 20 activity cliffs sorted by delta
        return sorted(cliffs, key=lambda x: x['delta'], reverse=True)[:20]

# ALTERNATIVE SIMPLER VERSION (if the above still has issues)
    def analyze_activity_cliffs_simple(self, activity_threshold=1.0):
        """
        Simplified activity cliff analysis using molecular similarity.
        """
        cliffs = []
        
        # Calculate molecular fingerprints for similarity comparison
        fps = []
        for idx, row in self.df.iterrows():
            mol = row['mol']
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                fps.append((row['mol_id'], fp, row['affinity'], mol))
            except:
                continue
        
        # Compare all pairs
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                id1, fp1, act1, mol1 = fps[i]
                id2, fp2, act2, mol2 = fps[j]
                
                # Calculate Tanimoto similarity
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                activity_delta = abs(act1 - act2)
                
                # Activity cliff: high similarity but large activity difference
                if similarity > 0.7 and activity_delta >= activity_threshold:
                    cliffs.append({
                        'mol1': mol1,
                        'mol2': mol2,
                        'act1': act1,
                        'act2': act2,
                        'delta': activity_delta,
                        'similarity': similarity,
                        'transform': f"Similar structures (Sim: {similarity:.2f})"
                    })
        
        return sorted(cliffs, key=lambda x: x['delta'], reverse=True)[:20]

# ==============================================================================
# 5. STREAMLIT APPLICATION CLASS
# ==============================================================================
class StreamlitSHAPAnalyzer:
    def __init__(self): self.initialize_session_state()
    def initialize_session_state(self):
        for key in ['analysis_data', 'uploaded_datasets', 'fragment_mappings']:
            if key not in st.session_state: st.session_state[key] = {}

    def run(self):
        st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
        st.markdown(CUSTOM_APP_CSS, unsafe_allow_html=True)
        if not HAS_RDKIT or not HAS_FINGERPRINT_FUNC: st.error("Missing dependencies."); return
        st.markdown(f'<h1 class="main-header">{PAGE_TITLE}</h1>', unsafe_allow_html=True)
        self._create_sidebar()
        if not st.session_state.analysis_data: self._show_getting_started()
        else: self._create_main_interface()

    def _create_sidebar(self):
        st.sidebar.header("📁 Data Upload")
        with st.sidebar.expander("Upload SHAP Analysis (.pkl)", True):
            uploaded_analysis = st.file_uploader("Select .pkl files", type=['pkl'], accept_multiple_files=True, label_visibility="collapsed")
            if uploaded_analysis: self._process_files(uploaded_analysis, 'analysis_data', pickle.load)
        with st.sidebar.expander("Upload Datasets (.csv)", True):
            uploaded_datasets = st.file_uploader("Select .csv files", type=['csv'], accept_multiple_files=True, label_visibility="collapsed")
            if uploaded_datasets: self._process_files(uploaded_datasets, 'uploaded_datasets', pd.read_csv, by_target=True)
        st.sidebar.markdown("---"); st.sidebar.header("⚙️ Analysis Settings")
        st.session_state.top_n_features = st.sidebar.slider("Top Features", 3, 20, 10)
        st.session_state.max_molecules = st.sidebar.slider("Molecules per Feature", 5, 50, 20)

    def _process_files(self, files, state_key, load_func, by_target=False):
        for f in files:
            key = f.name
            if by_target:
                target = "TYK2" if "TYK2" in f.name.upper() else "USP7" if "USP7" in f.name.upper() else None
                if target: key = target
            if key and key not in st.session_state[state_key]: st.session_state[state_key][key] = load_func(f)
        st.sidebar.success(f"Loaded {len(st.session_state[state_key])} files.")
        
    def _create_main_interface(self):
        tabs = ["📊 Overview", "📈 Evolution", "🧬 Fragments", "💊 Design", "📄 Figures", "🔬 Advanced Analytics"]
        tab_functions = [self._show_overview, self._show_evolution_analysis, self._show_chemical_fragments, self._show_drug_design, self._show_publication_figures, self._show_advanced_analytics]
        for tab, func in zip(st.tabs(tabs), tab_functions):
            with tab: func()

    def _show_overview(self):
        st.markdown('<h2 class="sub-header">Analysis Overview</h2>', unsafe_allow_html=True)
        st.write(f"**{len(st.session_state.analysis_data)}** analysis results and **{len(st.session_state.uploaded_datasets)}** datasets loaded.")
        overview_data = [{'File': f, 'Target': d.get('metadata', {}).get('target', 'N/A'), 'Protocol': d.get('metadata', {}).get('protocol', 'N/A')} for f, d in st.session_state.analysis_data.items()]
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True)

    def _show_evolution_analysis(self):
        st.markdown('<h2 class="sub-header">Feature Evolution Analysis</h2>', unsafe_allow_html=True)
        selected_file = st.selectbox("Select analysis file:", list(st.session_state.analysis_data.keys()))
        if not selected_file: return
        data = st.session_state.analysis_data[selected_file]; df = data.get('feature_evolution'); meta = data.get('metadata', {})
        protocol = meta.get('protocol', '')
        if df is not None:
            top_features = df.groupby('feature_index')['importance'].mean().nlargest(st.session_state.top_n_features).index
            plot_df = df[df['feature_index'].isin(top_features)]
            fig = px.line(plot_df, x='cycle', y='importance', color='feature_index', title=f"Top Feature Evolution: {meta.get('target', '')} - {protocol}", labels={'cycle': 'Cycle', 'importance': 'Mean |SHAP|'}, markers=True)
            if 'exploit-heavy' in protocol: fig.add_vrect(x0=-0.5, x1=3.5, fillcolor="blue", opacity=0.1, layer="below", line_width=0, annotation_text="Explore"); fig.add_vrect(x0=3.5, x1=10.5, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Exploit")
            elif 'explore-heavy' in protocol: fig.add_vrect(x0=-0.5, x1=7.5, fillcolor="blue", opacity=0.1, layer="below", line_width=0, annotation_text="Explore"); fig.add_vrect(x0=7.5, x1=10.5, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Exploit")
            st.plotly_chart(fig, use_container_width=True)

    def _show_chemical_fragments(self):
        st.markdown('<h2 class="sub-header">Chemical Fragment Analysis</h2>', unsafe_allow_html=True)
        selected_file = st.selectbox("Select analysis:", list(st.session_state.analysis_data.keys()), key="frag_select")
        if not selected_file: return
        meta = st.session_state.analysis_data[selected_file].get('metadata', {}); target, protocol = meta.get('target'), meta.get('protocol')
        if not target or target not in st.session_state.uploaded_datasets: st.error(f"Dataset for target '{target}' not found."); return
        if st.button("Analyze Fragments", key="frag_button", use_container_width=True):
            with st.spinner(f"Mapping fragments for {target}-{protocol}..."):
                mapper = ChemicalFragmentMapper(st.session_state.analysis_data[selected_file], st.session_state.uploaded_datasets[target], smiles_to_ecfp8)
                fragments = mapper.extract_fragments_for_features(target, protocol, st.session_state.top_n_features, st.session_state.max_molecules)
                st.session_state.fragment_mappings[f"{target}_{protocol}"] = (mapper, fragments)
        combo_key = f"{target}_{protocol}"
        if combo_key in st.session_state.fragment_mappings:
            mapper, fragments = st.session_state.fragment_mappings[combo_key]
            st.success(f"Found {len(fragments)} important features for {combo_key}.")
            for feature_idx, data in fragments.items():
                with st.expander(f"**Feature {feature_idx}** (Rank {data['rank']}, Avg. Importance: {data['average_importance']:.4f})"):
                    self._render_enhanced_feature_detail_tabs(selected_file, feature_idx, data, mapper)

    def _render_enhanced_feature_detail_tabs(self, selected_file, feature_idx, data, mapper):
        overview_tab, detail_tab, shap_tab, impact_tab = st.tabs(["Fragment Overview", "Fragment Details", "SHAP Contribution", "Feature Impact"])
        with overview_tab:
            col1, col2 = st.columns([0.4, 0.6])
            with col1:
                st.markdown("**All Common Fragments**"); st.info("Gallery of all frequent fragment structures for this feature.", icon="🧩")
                comprehensive_gallery = mapper.get_comprehensive_fragment_gallery(data['fragments'])
                if isinstance(comprehensive_gallery, plt.Figure): st.pyplot(comprehensive_gallery)
                elif comprehensive_gallery: st.image(comprehensive_gallery)
                else: st.warning("No common fragments identified.")
                st.markdown("**Affinity Distribution**"); affinity_fig = mapper.get_affinity_plot_fig(data)
                if affinity_fig: st.pyplot(affinity_fig)
            with col2:
                st.markdown("**Highlighted High-Affinity Molecules**"); st.info("Examples of where fragments appear in high-affinity molecules.", icon="🎯")
                for highlight_info in data['fragments']['molecule_highlights'][:3]:
                    mol_img = create_molecule_aura_image(highlight_info['mol'], highlight_info['highlight_atoms'], highlight_info['highlight_bonds'])
                    if mol_img: st.image(mol_img, caption=f"Affinity: {highlight_info['parent_info']['affinity']:.2f}")
        with detail_tab:
            st.markdown(f"### Detailed View: Most Common Fragment"); st.info("This shows the most common fragment isolated and then examples of where it appears.", icon="🔬")
            isolated_img, parent_images = mapper.get_integrated_fragment_display(data['fragments'])
            if isolated_img and parent_images:
                st.markdown("#### Isolated Fragment Structure"); col1, col2, col3 = st.columns([1, 2, 1])
                with col2: st.image(isolated_img, caption=f"Most Common Fragment: {data['fragments']['most_common'][0][0]}")
                st.markdown("#### This Fragment in High-Affinity Molecules"); st.markdown("*The same fragment structure highlighted in its parent molecules:*")
                cols = st.columns(len(parent_images) if parent_images else 1)
                for i, (col, p_info) in enumerate(zip(cols, parent_images)):
                    with col: st.image(p_info['image'], caption=f"Affinity: {p_info['affinity']:.2f}", use_column_width=True)
            else: st.warning("Could not generate detailed fragment analysis. This usually means no single fragment was consistently identified for this feature.")
        
        cycle_results = st.session_state.analysis_data[selected_file].get('cycle_results', {});
        if not cycle_results: st.warning("Raw `cycle_results` with SHAP values not found."); return
        with shap_tab:
            st.markdown(f"### Overall SHAP Contribution"); st.info("SHAP summary plot for all features in a given cycle.", icon="📈")
            cycle_shap = st.selectbox("Inspect SHAP values from cycle:", list(cycle_results.keys()), key=f"cycle_select_shap_{feature_idx}")
            if cycle_shap is not None:
                c_data = cycle_results[cycle_shap]
                shap_vals = shap.Explanation(values=c_data['shap_values'], base_values=c_data['shap_values'].mean(), data=c_data['test_data'], feature_names=[f"F{i}" for i in range(c_data['test_data'].shape[1])])
                fig, ax = plt.subplots(); shap.summary_plot(shap_vals, plot_type='dot', show=False, max_display=15); st.pyplot(fig)
        with impact_tab:
            st.markdown(f"### Impact of Feature {feature_idx}"); st.info(f"Shows which molecules were most affected by this specific feature.", icon="🎯")
            cycle_impact = st.selectbox("Inspect Feature Impact from cycle:", list(cycle_results.keys()), key=f"cycle_select_impact_{feature_idx}")
            if cycle_impact is not None:
                c_data = cycle_results[cycle_impact]
                feat_shap_vals = c_data['shap_values'][:, feature_idx]; sorted_idx = np.argsort(np.abs(feat_shap_vals))[::-1][:10]
                impact_df = pd.DataFrame({'Molecule Index in Test Set': sorted_idx, 'SHAP Value': feat_shap_vals[sorted_idx]})
                fig = px.bar(impact_df, x='SHAP Value', y='Molecule Index in Test Set', orientation='h', title=f"Top 10 Molecules Most Impacted by Feature {feature_idx}", color='SHAP Value', color_continuous_scale='RdBu_r')
                fig.update_layout(yaxis={'categoryorder':'total ascending'}); st.plotly_chart(fig, use_container_width=True)

    def _show_drug_design(self):
        st.markdown('<h2 class="sub-header">Molecular Design Templates</h2>', unsafe_allow_html=True)
        st.info("Generates synthesizable molecular scaffolds based on important chemical fragments.")
        target_to_design = st.selectbox("Select target:", list(st.session_state.uploaded_datasets.keys()), key="design_select")
        if st.button("Generate Design Templates", key="design_button", use_container_width=True):
            generator = MolecularDesignTemplateGenerator()
            scaffolds = generator.generate_scaffolds(target_to_design)
            if not scaffolds: st.warning("No design templates available."); return
            st.markdown(f"### Proposed Scaffolds for {target_to_design}"); st.image(generator.get_scaffold_grid_image(scaffolds))
            scaffold_data = [{'Scaffold': d['name'], 'SMILES': d['smiles'], **generator.calculate_properties(d['smiles'])} for n, d in scaffolds.items()]
            st.dataframe(pd.DataFrame(scaffold_data).round(2), use_container_width=True)

    def _show_publication_figures(self):
        st.markdown('<h2 class="sub-header">Publication-Ready Figures</h2>', unsafe_allow_html=True)
        st.info("Generates high-quality figures summarizing key findings.")
        if not st.session_state.fragment_mappings: st.warning("Run 'Chemical Fragment Analysis' first."); return
        combo_key = st.selectbox("Select analysis for figure:", list(st.session_state.fragment_mappings.keys()), key="pub_fig_select")
        if st.button("Generate Main Figure", key="pub_fig_button", use_container_width=True):
            target, protocol = combo_key.split('_', 1)
            generator = PublicationFigureGenerator(st.session_state.fragment_mappings, st.session_state.uploaded_datasets)
            fig = generator.create_main_figure(target, protocol)
            if fig:
                st.pyplot(fig); buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight'); st.download_button("Download Figure", buf.getvalue(), f"{combo_key}_key_features.png", "image/png")

    def _show_advanced_analytics(self):
        st.markdown('<h2 class="sub-header">Advanced Analytics & Insights</h2>', unsafe_allow_html=True)
        st.write("This section provides high-level comparisons across all loaded experimental conditions.")
        analysis_type = st.selectbox("Choose an analysis:", ["Cross-Target Feature Heatmap", "Feature Stability Analysis", "Chemical Pattern Recognition", "Structure-Activity Relationship (SAR)"])
        if analysis_type == "Cross-Target Feature Heatmap": self.create_cross_target_heatmap()
        elif analysis_type == "Feature Stability Analysis": self.analyze_feature_stability()
        elif analysis_type == "Chemical Pattern Recognition": self.identify_chemical_patterns()
        elif analysis_type == "Structure-Activity Relationship (SAR)": self.perform_sar_analysis()

    def create_cross_target_heatmap(self):
        st.info("This heatmap compares the normalized importance of key features across all loaded experiments.", icon="🗺️")
        if len(st.session_state.analysis_data) < 2: st.warning("Upload at least two analysis files for comparison."); return
        if st.button("Generate Heatmap", key="heatmap_btn", use_container_width=True):
            with st.spinner("Generating cross-target heatmap..."):
                all_importances = []
                for filename, data in st.session_state.analysis_data.items():
                    meta = data.get('metadata', {}); df = data.get('feature_evolution')
                    if df is not None and meta:
                        combo_key = f"{meta.get('target', 'T')}_{meta.get('protocol', 'P')}"
                        mean_importance = df.groupby('feature_index')['importance'].mean().reset_index()
                        mean_importance['combination'] = combo_key; all_importances.append(mean_importance)
                if not all_importances: st.error("Could not extract importance data."); return
                full_df = pd.concat(all_importances, ignore_index=True)
                top_features = set(full_df.groupby('combination').apply(lambda x: x.nlargest(20, 'importance')['feature_index']).explode())
                pivot_df = full_df[full_df['feature_index'].isin(top_features)].pivot_table(index='feature_index', columns='combination', values='importance', fill_value=0)
                norm_pivot_df = pivot_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0)
                clustergrid = sns.clustermap(norm_pivot_df, cmap='viridis', figsize=(12, max(10, len(top_features) * 0.3)), dendrogram_ratio=(0.1, 0.2))
                st.pyplot(clustergrid.fig)
                buf = io.BytesIO(); clustergrid.savefig(buf, format="png", dpi=300, bbox_inches='tight'); st.download_button("Download Heatmap", buf.getvalue(), "cross_target_heatmap.png", "image/png")

    def analyze_feature_stability(self):
        st.info("This analysis reveals which features are consistently important across active learning cycles.", icon="⚓")
        selected_file = st.selectbox("Select an experiment to analyze for stability:", list(st.session_state.analysis_data.keys()), key="stability_select")
        if not selected_file: return
        data = st.session_state.analysis_data[selected_file]; df = data.get('feature_evolution')
        if df is None: st.warning("Selected file does not contain `feature_evolution` data."); return
        with st.spinner("Calculating feature stability..."):
            stability_df = df.groupby('feature_index')['importance'].agg(['mean', 'std']).reset_index()
            stability_df['cv'] = (stability_df['std'] / stability_df['mean']).fillna(0)
            st.markdown("### Stability vs. Importance Plot"); fig, ax = plt.subplots(figsize=(10, 7))
            sns.scatterplot(data=stability_df, x='mean', y='cv', ax=ax, alpha=0.6, s=50)
            ax.set_xlabel("Mean Importance (Impact)"); ax.set_ylabel("Coefficient of Variation (Instability)"); ax.set_title("Feature Stability vs. Overall Importance")
            ax.axhline(stability_df['cv'].median(), ls='--', color='gray'); ax.axvline(stability_df['mean'].median(), ls='--', color='gray')
            st.pyplot(fig)
            st.markdown("### Ranked Feature Lists"); col1, col2 = st.columns(2)
            with col1: st.write("**Most Stable Features (Lowest CV)**"); st.dataframe(stability_df.sort_values('cv').head(15).round(4))
            with col2: st.write("**Most Important Features (Highest Mean)**"); st.dataframe(stability_df.sort_values('mean', ascending=False).head(15).round(4))

    def identify_chemical_patterns(self):
        st.info("This tool detects chemical motifs in the most stable and important features for a selected experiment.", icon="🔬")
        if not st.session_state.fragment_mappings: st.warning("Run 'Chemical Fragment Analysis' first."); return
        combo_key = st.selectbox("Select fragment analysis to recognize patterns from:", list(st.session_state.fragment_mappings.keys()), key="pattern_select")
        if not combo_key: return
        mapper, fragments = st.session_state.fragment_mappings[combo_key]
        all_top_fragments = [frag[0] for data in fragments.values() for frag in data.get('fragments', {}).get('most_common', [])]
        if not all_top_fragments: st.warning("No common fragments found in the selected analysis."); return
        patterns = {'Halogens': 0, 'Aromatic': 0, 'Nitrogen': 0, 'Oxygen': 0, 'Sulfur': 0}
        for smiles in all_top_fragments:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: continue
            if any(atom.GetSymbol() in ['F', 'Cl', 'Br', 'I'] for atom in mol.GetAtoms()): patterns['Halogens'] += 1
            if any(atom.GetIsAromatic() for atom in mol.GetAtoms()): patterns['Aromatic'] += 1
            if 'N' in smiles: patterns['Nitrogen'] += 1;
            if 'O' in smiles: patterns['Oxygen'] += 1;
            if 'S' in smiles: patterns['Sulfur'] += 1
        pattern_df = pd.DataFrame.from_dict(patterns, orient='index', columns=['Count'])
        pattern_df['Percentage'] = (pattern_df['Count'] / len(all_top_fragments) * 100).round(1)
        st.markdown(f"### Chemical Patterns in Top Features for **{combo_key}**")
        fig = px.bar(pattern_df, y='Percentage', x=pattern_df.index, title="Prevalence of Key Chemical Patterns", text='Percentage')
        fig.update_traces(texttemplate='%{text}%', textposition='outside'); st.plotly_chart(fig, use_container_width=True)

    def perform_sar_analysis(self):
        st.info("This section performs classic medicinal chemistry analyses to find activity cliffs and R-group effects.", icon="💊")
        target = st.selectbox("Select dataset for SAR analysis:", list(st.session_state.uploaded_datasets.keys()), key="sar_target_select")
        if not target: return
        dataset_df = st.session_state.uploaded_datasets[target]; analyzer = SARAnalyzer(dataset_df)
        st.markdown("#### R-Group Decomposition");
        if st.button("Analyze R-Groups", key="rgroup_btn"):
            with st.spinner("Finding common scaffold and decomposing R-groups..."): st.session_state[f'r_group_data_{target}'] = analyzer.analyze_r_groups()
        if f'r_group_data_{target}' in st.session_state:
            core, r_group_df = st.session_state[f'r_group_data_{target}']
            st.write("**Most Common Murcko Scaffold:**"); st.image(Draw.MolToImage(core))
            st.dataframe(r_group_df.drop(columns=['mol', 'Core'], errors='ignore'))
        st.markdown("#### Activity Cliff Identification")
        activity_threshold = st.slider("Minimum Activity Change for Cliff", 0.5, 5.0, 1.0, 0.1)
        if st.button("Identify Activity Cliffs", key="cliff_btn"):
            with st.spinner(f"Searching for activity cliffs..."): st.session_state[f'cliffs_{target}'] = analyzer.analyze_activity_cliffs(activity_threshold)
        if f'cliffs_{target}' in st.session_state:
            cliffs = st.session_state[f'cliffs_{target}']
            st.success(f"Found {len(cliffs)} activity cliffs.")
            for i, cliff in enumerate(cliffs[:5]):
                st.markdown(f"**Cliff #{i+1} | Transformation: `{cliff['transform']}` | ΔActivity: {cliff['delta']:.2f}**")
                col1, col2 = st.columns(2)
                with col1: st.image(Draw.MolToImage(cliff['mol1']), caption=f"Activity: {cliff['act1']:.2f}")
                with col2: st.image(Draw.MolToImage(cliff['mol2']), caption=f"Activity: {cliff['act2']:.2f}")
                st.divider()

    def _show_getting_started(self):
        st.markdown('<h2 class="sub-header">🚀 Getting Started</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h4>Welcome to the SHAP-Guided Drug Discovery Platform!</h4>
            <ol>
                <li><b>Upload SHAP analysis files:</b> The <code>.pkl</code> files from your pipeline.</li>
                <li><b>Upload molecule datasets:</b> The <code>.csv</code> files with SMILES and affinity data.</li>
                <li><b>Run Analysis:</b> Navigate the tabs and click the buttons to run each analysis step. Results from one tab (like fragment mapping) are used in subsequent tabs.</li>
            </ol>
        </div>""", unsafe_allow_html=True)

# ==============================================================================
# 6. SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    app = StreamlitSHAPAnalyzer()
    app.run()
