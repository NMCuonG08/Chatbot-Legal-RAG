import React, { createContext, useContext, useEffect, useState } from 'react';
import { User, AuthState } from '../types';
import { fetchMeApi, loginApi, registerApi } from '../services/api';

interface AuthContextType extends AuthState {
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string, role?: string) => Promise<void>;
  logout: () => void;
  showAuthModal: boolean;
  setShowAuthModal: (show: boolean) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('legal_rag_jwt_token'));
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [showAuthModal, setShowAuthModal] = useState<boolean>(false);

  useEffect(() => {
    async function initAuth() {
      if (token) {
        try {
          const userData = await fetchMeApi();
          setUser(userData);
        } catch (err) {
          console.warn('Invalid token or session expired');
          localStorage.removeItem('legal_rag_jwt_token');
          setToken(null);
          setUser(null);
        }
      }
      setIsLoading(false);
    }
    initAuth();
  }, [token]);

  const login = async (username: string, password: string) => {
    const res = await loginApi(username, password);
    localStorage.setItem('legal_rag_jwt_token', res.access_token);
    setToken(res.access_token);
    setUser(res.user);
    setShowAuthModal(false);
  };

  const register = async (username: string, password: string, role: string = 'user') => {
    const res = await registerApi(username, password, role);
    localStorage.setItem('legal_rag_jwt_token', res.access_token);
    setToken(res.access_token);
    setUser(res.user);
    setShowAuthModal(false);
  };

  const logout = () => {
    localStorage.removeItem('legal_rag_jwt_token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        isLoading,
        login,
        register,
        logout,
        showAuthModal,
        setShowAuthModal,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
