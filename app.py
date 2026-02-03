with col1:
        st.write("**Distribusi Kategori Udara**")
        # Menentukan urutan kategori agar rapi dari Baik ke Berbahaya
        kategori_order = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT', 'BERBAHAYA']
        
        fig1, ax1 = plt.subplots()
        # Menggunakan data yang ada di dataset, tapi diurutkan berdasarkan standar
        sns.countplot(
            data=st.session_state.df_full, 
            x='categori', 
            palette='viridis', 
            order=[c for c in kategori_order if c in st.session_state.df_full['categori'].unique()],
            ax=ax1
        )
        plt.xticks(rotation=45)
        st.pyplot(fig1)
