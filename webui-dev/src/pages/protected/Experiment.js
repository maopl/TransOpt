import { useEffect } from 'react'
import { useDispatch } from 'react-redux'
import { setPageTitle } from '../../features/common/headerSlice'
import Experiment from '../../features/experiment/index'
import Seldata from "../../features/seldata/index"


function InternalPage(){
    const dispatch = useDispatch()

    useEffect(() => {
        dispatch(setPageTitle({ title : "Experiment" }))
      }, [])


    return(
       
            <div >
              <Experiment />
        
                {/* <Seldata /> */}
              
               
            </div>
          
          );
}

export default InternalPage

