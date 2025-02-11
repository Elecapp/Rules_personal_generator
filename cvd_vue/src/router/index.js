import Vue from 'vue';
import Router from 'vue-router';
import VesselsNeighborhood from '../components/VesselsNeighborhood';
import VesselsRules from '../components/VesselsRules';
import LandingPage from "../components/LandingPage.vue";

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/',
      name: 'LandingPage',
      component: LandingPage,
    },
    {
      path: '/vessels-neighborhood',
      name: 'VesselsNeighborhood',
      component: VesselsNeighborhood,
    },
    {
      path: '/vessels-rules',
      name: 'VesselsRules',
      component: VesselsRules,
    },
  ],
});
