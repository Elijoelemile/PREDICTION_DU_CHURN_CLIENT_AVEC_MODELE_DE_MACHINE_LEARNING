import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Prédiction Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .feature-importance {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ChurnDashboard:
    def __init__(self):
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Charger et préparer les données"""
        # Dans un cas réel, vous chargeriez vos données ici
        # Pour l'exemple, nous allons créer des données synthétiques
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(50, 8000, n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples)
        })
        
        # Créer une variable cible synthétique
        churn_proba = (
            (data['tenure'] < 12) * 0.3 +
            (data['Contract'] == 'Month-to-month') * 0.4 +
            (data['MonthlyCharges'] > 70) * 0.2 +
            (data['OnlineSecurity'] == 'No') * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        data['Churn'] = (churn_proba > 0.5).astype(int)
        data['Churn'] = data['Churn'].map({0: 'No', 1: 'Yes'})
        
        self.data = data
        return data
    
    def preprocess_data(self):
        """Prétraiter les données pour la modélisation"""
        df = self.data.copy()
        
        # Encodage des variables catégorielles
        categorical_cols = ['Contract', 'InternetService', 'OnlineSecurity', 
                           'TechSupport', 'PaymentMethod', 'Partner', 'Dependents']
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Séparation features/target
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return X, y
    
    def train_model(self):
        """Entraîner le modèle"""
        X, y = self.preprocess_data()
        
        # Utiliser Random Forest comme modèle principal
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        return self.model
    
    def plot_churn_distribution(self):
        """Diagramme de distribution du churn"""
        fig = px.pie(self.data, names='Churn', title='Distribution du Churn Client',
                    color='Churn', color_discrete_map={'Yes': '#FF4B4B', 'No': '#0068C9'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    def plot_tenure_vs_churn(self):
        """Ancienneté vs Churn"""
        fig = px.histogram(self.data, x='tenure', color='Churn', 
                          title='Distribution de l\'Ancienneté par Statut Churn',
                          barmode='overlay', opacity=0.7,
                          color_discrete_map={'Yes': '#FF4B4B', 'No': '#0068C9'})
        fig.update_layout(xaxis_title='Ancienneté (mois)', yaxis_title='Nombre de Clients')
        return fig
    
    def plot_contract_churn(self):
        """Churn par type de contrat"""
        contract_churn = pd.crosstab(self.data['Contract'], self.data['Churn'], normalize='index') * 100
        fig = px.bar(contract_churn, barmode='group', 
                    title='Taux de Churn par Type de Contrat (%)',
                    color_discrete_map={'Yes': '#FF4B4B', 'No': '#0068C9'})
        fig.update_layout(xaxis_title='Type de Contrat', yaxis_title='Pourcentage (%)')
        return fig
    
    def plot_monthly_charges(self):
        """Charges mensuelles vs Churn"""
        fig = px.box(self.data, x='Churn', y='MonthlyCharges', 
                    title='Distribution des Charges Mensuelles par Statut Churn',
                    color='Churn', color_discrete_map={'Yes': '#FF4B4B', 'No': '#0068C9'})
        return fig
    
    def plot_feature_importance(self):
        """Importance des features"""
        if self.model is None:
            self.train_model()
        
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(feature_importance.tail(15), x='importance', y='feature',
                    title='Top 15 des Variables les Plus Importantes',
                    orientation='h')
        fig.update_layout(yaxis_title='Variables', xaxis_title='Importance')
        return fig
    
    def plot_confusion_matrix(self):
        """Matrice de confusion"""
        if self.model is None:
            self.train_model()
        
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       labels=dict(x="Prédit", y="Réel", color="Count"),
                       x=['No', 'Yes'], y=['No', 'Yes'],
                       title='Matrice de Confusion')
        return fig
    
    def plot_roc_curve(self):
        """Courbe ROC"""
        if self.model is None:
            self.train_model()
        
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba, pos_label='Yes')
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                               name=f'ROC curve (AUC = {auc_score:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                               name='Random', line=dict(dash='dash')))
        
        fig.update_layout(title='Courbe ROC',
                         xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
        return fig
    
    def get_business_metrics(self):
        """Calculer les métriques business"""
        total_clients = len(self.data)
        churn_rate = (self.data['Churn'] == 'Yes').mean() * 100
        avg_tenure = self.data['tenure'].mean()
        avg_monthly_charge = self.data['MonthlyCharges'].mean()
        
        # Taux de churn par contrat
        contract_churn = self.data.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        )
        
        return {
            'total_clients': total_clients,
            'churn_rate': churn_rate,
            'avg_tenure': avg_tenure,
            'avg_monthly_charge': avg_monthly_charge,
            'contract_churn': contract_churn
        }

def main():
    st.markdown('<h1 class="main-header">📊 Dashboard Prédiction du Churn Client</h1>', 
                unsafe_allow_html=True)
    
    # Initialiser le dashboard
    dashboard = ChurnDashboard()
    data = dashboard.load_data()
    
    # Sidebar
    st.sidebar.title("🔧 Paramètres")
    st.sidebar.markdown("### Navigation")
    
    sections = [
        "📈 Vue d'ensemble",
        "🔍 Analyse Exploratoire", 
        "🤖 Modélisation ML",
        "💡 Insights Business",
        "🎯 Prédictions"
    ]
    
    selected_section = st.sidebar.radio("Sélectionnez une section:", sections)
    
    # Métriques rapides dans la sidebar
    st.sidebar.markdown("### Métriques Clés")
    metrics = dashboard.get_business_metrics()
    
    st.sidebar.metric("Clients Totaux", f"{metrics['total_clients']:,}")
    st.sidebar.metric("Taux de Churn", f"{metrics['churn_rate']:.1f}%")
    st.sidebar.metric("Ancienneté Moyenne", f"{metrics['avg_tenure']:.1f} mois")
    st.sidebar.metric("Charge Mensuelle Moyenne", f"${metrics['avg_monthly_charge']:.2f}")
    
    # Section 1: Vue d'ensemble
    if selected_section == "📈 Vue d'ensemble":
        st.header("📈 Vue d'Ensemble des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(dashboard.plot_churn_distribution(), use_container_width=True)
        
        with col2:
            # Aperçu des données
            st.subheader("Aperçu des Données")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Statistiques descriptives
            st.subheader("Statistiques Descriptives")
            st.dataframe(data.describe(), use_container_width=True)
    
    # Section 2: Analyse Exploratoire
    elif selected_section == "🔍 Analyse Exploratoire":
        st.header("🔍 Analyse Exploratoire")
        
        st.subheader("Distribution de l'Ancienneté")
        st.plotly_chart(dashboard.plot_tenure_vs_churn(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(dashboard.plot_contract_churn(), use_container_width=True)
        
        with col2:
            st.plotly_chart(dashboard.plot_monthly_charges(), use_container_width=True)
        
        # Analyse des services
        st.subheader("Analyse des Services")
        service_cols = ['InternetService', 'OnlineSecurity', 'TechSupport']
        
        for service in service_cols:
            fig = px.histogram(data, x=service, color='Churn', 
                             title=f'Churn par {service}',
                             barmode='group',
                             color_discrete_map={'Yes': '#FF4B4B', 'No': '#0068C9'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Section 3: Modélisation ML
    elif selected_section == "🤖 Modélisation ML":
        st.header("🤖 Modélisation Machine Learning")
        
        with st.spinner("Entraînement du modèle en cours..."):
            dashboard.train_model()
        
        st.success("Modèle entraîné avec succès!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(dashboard.plot_feature_importance(), use_container_width=True)
            st.plotly_chart(dashboard.plot_confusion_matrix(), use_container_width=True)
        
        with col2:
            st.plotly_chart(dashboard.plot_roc_curve(), use_container_width=True)
            
            # Métriques de performance
            y_pred = dashboard.model.predict(dashboard.X_test)
            report = classification_report(dashboard.y_test, y_pred, output_dict=True)
            
            st.subheader("Rapport de Classification")
            st.json({
                'Accuracy': f"{report['accuracy']:.3f}",
                'Precision (Yes)': f"{report['Yes']['precision']:.3f}",
                'Recall (Yes)': f"{report['Yes']['recall']:.3f}",
                'F1-Score (Yes)': f"{report['Yes']['f1-score']:.3f}"
            })
    
    # Section 4: Insights Business
    elif selected_section == "💡 Insights Business":
        st.header("💡 Insights et Recommandations Business")
        
        metrics = dashboard.get_business_metrics()
        
        st.subheader("🎯 Segments à Haut Risque")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Clients Month-to-month", 
                     f"{len(data[data['Contract'] == 'Month-to-month'])}",
                     "Segment le plus risqué")
        
        with col2:
            st.metric("Ancienneté < 6 mois", 
                     f"{len(data[data['tenure'] < 6])}",
                     "Fort risque de départ")
        
        with col3:
            st.metric("Charges > $70", 
                     f"{len(data[data['MonthlyCharges'] > 70])}",
                     "Sensibilité au prix")
        
        st.subheader("📋 Plan d'Action Recommandé")
        
        actions = [
            {
                "priorité": "🔴 Haute",
                "action": "Programme de fidélisation clients Month-to-month",
                "cible": "Contrat mensuel + ancienneté < 12 mois",
                "impact": "Réduction churn 15-20%"
            },
            {
                "priorité": "🟠 Moyenne", 
                "action": "Offres personnalisées clients haut risque",
                "cible": "Ancienneté < 6 mois + charges élevées",
                "impact": "Amélioration rétention 10%"
            },
            {
                "priorité": "🟢 Basse",
                "action": "Renforcement support technique",
                "cible": "Clients sans OnlineSecurity/TechSupport",
                "impact": "Réduction churn 5-8%"
            }
        ]
        
        for action in actions:
            with st.expander(f"{action['priorité']} {action['action']}"):
                st.write(f"**Cible:** {action['cible']}")
                st.write(f"**Impact estimé:** {action['impact']}")
    
    # Section 5: Prédictions
    elif selected_section == "🎯 Prédictions":
        st.header("🎯 Prédire le Risque de Churn")
        
        st.subheader("Simulateur de Prédiction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.slider("Ancienneté (mois)", 1, 72, 12)
            monthly_charges = st.slider("Charges Mensuelles ($)", 20, 120, 65)
            contract = st.selectbox("Type de Contrat", 
                                  ['Month-to-month', 'One year', 'Two year'])
            internet_service = st.selectbox("Service Internet", 
                                          ['DSL', 'Fiber optic', 'No'])
        
        with col2:
            online_security = st.selectbox("Sécurité En Ligne", 
                                         ['Yes', 'No', 'No internet service'])
            tech_support = st.selectbox("Support Technique", 
                                      ['Yes', 'No', 'No internet service'])
            payment_method = st.selectbox("Méthode de Paiement", 
                                        ['Electronic check', 'Mailed check', 
                                         'Bank transfer', 'Credit card'])
            senior_citizen = st.radio("Senior Citizen", [0, 1])
        
        if st.button("🔮 Prédire le Risque de Churn"):
            # Créer un dataframe avec les inputs
            input_data = pd.DataFrame({
                'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [tenure * monthly_charges],
                'Contract': [contract],
                'InternetService': [internet_service],
                'OnlineSecurity': [online_security],
                'TechSupport': [tech_support],
                'PaymentMethod': [payment_method],
                'SeniorCitizen': [senior_citizen],
                'Partner': ['Yes'],  # Valeurs par défaut pour la démo
                'Dependents': ['No']
            })
            
            # Prétraiter et faire la prédiction
            try:
                # Encodage similaire à l'entraînement
                categorical_cols = ['Contract', 'InternetService', 'OnlineSecurity', 
                                  'TechSupport', 'PaymentMethod', 'Partner', 'Dependents']
                
                input_encoded = pd.get_dummies(input_data, columns=categorical_cols)
                
                # Aligner les colonnes avec l'entraînement
                trained_columns = dashboard.X_train.columns
                for col in trained_columns:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                
                input_encoded = input_encoded[trained_columns]
                
                # Prédiction
                prediction = dashboard.model.predict(input_encoded)[0]
                probability = dashboard.model.predict_proba(input_encoded)[0]
                
                # Affichage des résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 'Yes':
                        st.error(f"🚨 Risque Élevé: {probability[1]*100:.1f}% de churn")
                    else:
                        st.success(f"✅ Faible Risque: {probability[0]*100:.1f}% de rétention")
                
                with col2:
                    # Jauge de risque
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = probability[1] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Score de Risque"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90}}))
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommandations personnalisées
                st.subheader("💡 Recommandations Personnalisées")
                
                recommendations = []
                if contract == 'Month-to-month':
                    recommendations.append("💡 **Offrir un contrat annuel** avec réduction de 10%")
                if tenure < 6:
                    recommendations.append("👋 **Programme bienvenue** avec support dédié")
                if monthly_charges > 70:
                    recommendations.append("💰 **Analyser optimisation** des services")
                if online_security == 'No':
                    recommendations.append("🛡️ **Proposer sécurité en ligne** gratuite 3 mois")
                
                for rec in recommendations:
                    st.write(rec)
                    
            except Exception as e:
                st.error(f"Erreur dans la prédiction: {str(e)}")
                st.info("Assurez-vous que le modèle est entraîné dans la section Modélisation")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Dashboard développé pour l'analyse et la prédiction du churn client** • "
        "Utilisez les insights pour améliorer votre stratégie de rétention!"
    )

if __name__ == "__main__":
    main()